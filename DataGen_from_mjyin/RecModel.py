import torch.nn as nn
import torch
import logging
from typing import Optional, Union
from transformers import Qwen3ForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss

def ForRecLoss(
    logits: torch.Tensor,  # These are now hidden states, not raw scores for all vocab
    labels: torch.Tensor,
    vocab_size: int,
    item_embeddings: torch.Tensor, # Added: embeddings for all items (vocab_size, embedding_dim)
    num_negative_samples: int = 5, # Added: Number of negative samples
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float() # These are the hidden states from the model

    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    _, _, hidden_size = logits.shape

    # Reshape hidden states for processing
    # Each position in the sequence will predict an item
    # So, we flatten (batch_size * seq_len, hidden_size)
    hidden_states_flat = logits.view(-1, hidden_size)
    shift_labels_flat = shift_labels.view(-1)

    # Filter out ignored labels (e.g., padding tokens)
    valid_indices = (shift_labels_flat != ignore_index).nonzero(as_tuple=True)[0]
    if valid_indices.numel() == 0:
        return torch.tensor(0.0, device=logits.device) # No valid labels to compute loss

    hidden_states_valid = hidden_states_flat[valid_indices]
    target_labels_valid = shift_labels_flat[valid_indices]

    # --- Sampled Softmax Implementation ---

    # 1. Get scores for positive items
    # Dot product between hidden states and positive item embeddings
    # (num_valid_items, hidden_size) @ (hidden_size, 1) -> (num_valid_items, 1)
    positive_item_embeddings = item_embeddings[target_labels_valid]
    positive_scores = torch.sum(hidden_states_valid * positive_item_embeddings, dim=1, keepdim=True)

    # 2. Sample negative items
    with torch.no_grad():
        negative_samples = torch.randint(
            4, vocab_size, size=(target_labels_valid.size(0), num_negative_samples), device=logits.device
        )
    
    # 3. Get scores for negative items
    # (num_valid_items, num_negative_samples, hidden_size)
    negative_item_embeddings = item_embeddings[negative_samples]
    negative_scores = torch.sum(hidden_states_valid.unsqueeze(1) * negative_item_embeddings, dim=2)
    # (num_valid_items, num_negative_samples)

    # 4. Combine positive and negative scores
    # Scores for (positive_item, negative_sample_1, ..., negative_sample_N)
    # Target label for this combined tensor will be 0 (index of the positive item)
    all_scores = torch.cat([positive_scores, negative_scores], dim=1) # (num_valid_items, 1 + num_negative_samples)

    # The target for `cross_entropy` is 0, as the positive item is at index 0
    target_for_sampled_softmax = torch.zeros(
        all_scores.size(0), dtype=torch.long, device=all_scores.device
    )

    # Compute sampled softmax loss (equivalent to cross-entropy on sampled scores)
    loss = nn.functional.cross_entropy(all_scores, target_for_sampled_softmax, reduction="mean")

    return loss

class RecModel(Qwen3ForCausalLM):
    """
    This RecModel is modified from the original Qwen3ForCausalLM to adapt it for large vocab size in recommendation tasks.
    """

    @property
    def loss_function(self):
        return ForRecLoss

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
        # **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = hidden_states[:, slice_indices, :]

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, # These are the hidden states, not final item scores
                labels=labels,
                vocab_size=self.config.vocab_size,
                item_embeddings=self.get_output_embeddings().weight, # Pass the actual item embeddings
                num_negative_samples=256,
                **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
