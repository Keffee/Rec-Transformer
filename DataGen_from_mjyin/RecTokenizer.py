
import os # For creating directories
import logging

logging.basicConfig(level=logging.INFO)

from datasets import Dataset

from datasets import Dataset # Assuming you have a Hugging Face Dataset object
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing # For BOS/EOS
from transformers import PreTrainedTokenizerFast, TrainingArguments

def create_rec_tokenizer(dataset: Dataset, training_args: TrainingArguments) -> PreTrainedTokenizerFast:
    """
    Initializes a custom tokenizer for recommendation sequences based on a Hugging Face Dataset.
    This tokenizer uses the `tokenizers` library directly for precise vocabulary control.

    Args:
        dataset (datasets.Dataset): The Hugging Face dataset with format [user_id, item_sequence].
                                    item_sequence is a list of item IDs (integers).
        tokenizer_output_path (str): Directory where the tokenizer files will be saved.
                                     Defaults to "./rec_tokenizer".

    Returns:
        transformers.PreTrainedTokenizerFast: The initialized tokenizer with a custom vocabulary.
    """

    # 1. Extract all unique item IDs from the dataset
    all_item_ids = set()
    for row in dataset:
        if 'item_sequence' in row and isinstance(row['item_sequence'], list):
            for item_id in row['item_sequence']:
                all_item_ids.add(item_id)
        else:
            logging.warning(f"Warning: 'item_sequence' not found or not a list in row: {row}")
            continue

    # Convert set to sorted list for consistent token IDs
    all_item_ids = sorted(list(all_item_ids))
    logging.info(f"Total unique item IDs found: {len(all_item_ids)}")

    # 2. Define special tokens
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

    # 3. Build vocabulary (vocab)
    # This is a "token string -> token_id integer" mapping
    vocab = {}
    
    # First, add special tokens with IDs starting from 0
    for i, token in enumerate(special_tokens):
        vocab[token] = i

    # Then, add your item IDs.
    # We convert integer item IDs to strings for the vocabulary keys.
    # Token IDs for items start immediately after the special tokens.
    offset = len(special_tokens)
    for i, item_id in enumerate(all_item_ids):
        vocab[str(item_id)] = i + offset

    logging.info(f"\nVocabulary example (first few special tokens):")
    logging.info({k: v for k, v in list(vocab.items())[:len(special_tokens)]})
    logging.info(f"Vocabulary example (first few item ID tokens):")
    if len(all_item_ids) > 0:
        logging.info({k: v for k, v in list(vocab.items())[len(special_tokens) : len(special_tokens) + min(4, len(all_item_ids))]})
    logging.info(f"\nTotal vocabulary size: {len(vocab)}")

    # --- Use the `tokenizers` library to build the tokenizer ---

    # 4. Initialize a `WordLevel` Tokenizer with our custom vocabulary
    # `vocab` is the token_string -> id mapping
    # `unk_token` specifies which token to use for unknown words
    custom_tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))

    # 5. Set Pre-tokenizer
    # `Whitespace` will split the input string by spaces.
    # This means your item ID sequences will need to be provided as space-separated strings
    # e.g., "[BOS] 101 42 8 [EOS]"
    custom_tokenizer.pre_tokenizer = Whitespace()

    # 6. (Optional but Recommended) Set post-processor to automatically add [BOS] and [EOS]
    # This is very useful for sequence models as it simplifies tokenization.
    # We define the template for a single sequence: `[BOS]` then the sequence `$A` then `[EOS]`.
    # `special_tokens` maps the special token strings to their IDs in our vocabulary.
    custom_tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", vocab["[BOS]"]),
            ("[EOS]", vocab["[EOS]"]),
        ],
    )

    # Set special token IDs directly in the tokenizer for consistency
    custom_tokenizer.add_special_tokens(special_tokens)
    custom_tokenizer.enable_padding(pad_id=vocab["[PAD]"], pad_token="[PAD]", direction="left")
    custom_tokenizer.enable_truncation(max_length=training_args.max_length, direction="left") # Example max length, adjust as needed

    # 7. Save the tokenizer file
    # Create the directory if it doesn't exist
    os.makedirs(training_args.output_dir, exist_ok=True)
    tokenizer_file_path = os.path.join(training_args.output_dir, "tokenizer.json")
    custom_tokenizer.save(tokenizer_file_path)

    logging.info(f"\nTokenizer configuration saved to {tokenizer_file_path}")

    # 8. Wrap the `tokenizers` object with `transformers.PreTrainedTokenizerFast`
    # This allows you to use it like any other Hugging Face tokenizer (e.g., for model input).
    # We need to pass the special token attributes that `PreTrainedTokenizerFast` expects.
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file_path,
        # Special tokens are defined directly here, mapping to our vocab IDs
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        # Ensure these IDs match your vocab definition
        pad_token_id=vocab["[PAD]"],
        unk_token_id=vocab["[UNK]"],
        bos_token_id=vocab["[BOS]"],
        eos_token_id=vocab["[EOS]"],
        # Add the full vocabulary for the .vocab property (optional, but good for inspection)
        vocab=vocab
    )

    logging.info(f"Hugging Face PreTrainedTokenizerFast created successfully.")
    logging.info(f"Final tokenizer vocab size: {len(hf_tokenizer.vocab)}")

    return hf_tokenizer
