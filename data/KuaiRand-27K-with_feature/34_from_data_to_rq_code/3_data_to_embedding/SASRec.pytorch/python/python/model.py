import numpy as np
import torch
import torch.nn.init as init

# PointWiseFeedForward 保持不变
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs # Residual connection
        return outputs

# --- 新模块：封装所有特征处理逻辑 (已修改) ---
class FeatureEnhancedEmbedding(torch.nn.Module):
    def __init__(self, item_num, args, feature_info):
        super(FeatureEnhancedEmbedding, self).__init__()
        self.use_features = (feature_info is not None)
        # self.use_features = False
        self.dev = args.device

        # 1. 基础的 Item ID 嵌入
        self.id_embedding = torch.nn.Embedding(item_num + 1, args.hidden_units, padding_idx=0)

        if self.use_features:
            # 将特征张量注册为 buffer (非模型参数，但随模型移动到 GPU/CPU)
            self.register_buffer('discrete_features', feature_info['discrete_features_tensor'])
            self.register_buffer('continuous_features', feature_info['continuous_features_tensor'])

            # 2. 离散特征的嵌入层
            self.discrete_embeddings = torch.nn.ModuleList([
                torch.nn.Embedding(cardinality, args.feature_emb_dim)
                for cardinality in feature_info['discrete_cardinalities']
            ])

            # 3. 连续特征的独立处理网络 (修改的核心部分)
            # 为每个连续特征创建一个独立的投影网络
            cont_count = feature_info['continuous_feature_count']
            self.continuous_projections = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(1, args.feature_emb_dim // 2), # 输入维度为1
                    torch.nn.LayerNorm(args.feature_emb_dim // 2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(args.feature_emb_dim // 2, args.feature_emb_dim)
                ) for _ in range(cont_count)
            ])

            # 4. 特征融合 MLP (修改了输入维度并添加了LayerNorm)
            # 总输入维度 = (离散特征数 + 连续特征数) * 特征嵌入维度
            total_feature_dim = (len(self.discrete_embeddings) + cont_count) * args.feature_emb_dim
            self.feature_mlp = torch.nn.Sequential(
                torch.nn.Linear(total_feature_dim, args.hidden_units),
                torch.nn.LayerNorm(args.hidden_units), # 添加层归一化
                torch.nn.ReLU(),
                torch.nn.Linear(args.hidden_units, args.hidden_units // 2)
                # 在最终输出前通常可以不加Norm和激活，让信息直接流动
            )

            # 5. 最终融合层 (ID嵌入 + 处理后的特征)
            final_input_dim = args.hidden_units + (args.hidden_units // 2)
            self.final_projection = torch.nn.Linear(final_input_dim, args.hidden_units)        # 在所有层定义之后，调用初始化方法
            self.final_norm = torch.nn.LayerNorm(args.hidden_units) 

    def forward(self, item_indices):
        # 获取基础 ID 嵌入
        id_embs = self.id_embedding(item_indices)

        if not self.use_features:
            return id_embs

        # ---- 特征处理流程 ----
        # 1. 查找原始特征
        # disc_feats_raw 的形状: (batch, seq_len, num_discrete_feats)
        disc_feats_raw = self.discrete_features[item_indices]
        # cont_feats_raw 的形状: (batch, seq_len, num_cont_feats)
        cont_feats_raw = self.continuous_features[item_indices]

        # 2. 处理离散特征
        embedded_disc_feats = []
        for i, emb_layer in enumerate(self.discrete_embeddings):
            embedded_disc_feats.append(emb_layer(disc_feats_raw[..., i]))
        
        # 3. 独立处理每个连续特征 (修改的核心部分)
        embedded_cont_feats = []
        for i, proj_layer in enumerate(self.continuous_projections):
            # 每次取出一个连续特征，并保持其最后一维为1以匹配Linear(1,...)
            # cont_feats_raw[..., i:i+1] 的形状: (batch, seq_len, 1)
            cont_feat_slice = cont_feats_raw[..., i:i+1]
            embedded_cont_feats.append(proj_layer(cont_feat_slice))

        # 4. 拼接所有特征嵌入并过 MLP
        # 现在连续特征和离散特征的嵌入列表直接拼接
        all_feature_embs = torch.cat(embedded_disc_feats + embedded_cont_feats, dim=-1)
        processed_features = self.feature_mlp(all_feature_embs)
        
        # 5. 拼接 ID 嵌入和处理后的特征
        final_concat = torch.cat([id_embs, processed_features], dim=-1)
        
        # 6. 最终投影
        final_embedding = self.final_projection(final_concat)
        
        return final_embedding

# --- 重构后的 SASRec 模型 ---
class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args, feature_info=None):
        super(SASRec, self).__init__()
        self.dev = args.device

        # 使用新的 FeatureEnhancedEmbedding 模块
        self.item_emb_module = FeatureEnhancedEmbedding(item_num, args, feature_info)
        
        # 引用原始的 id_embedding 以便进行 L2 正则化和初始化
        self.item_emb = self.item_emb_module.id_embedding
        
        self.pos_emb = torch.nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.attention_layers.append(torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate))
            self.forward_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(args.hidden_units, args.dropout_rate))

    def get_all_item_embeddings(self):
        """为保存 .npy 文件提供接口"""
        all_item_ids = torch.arange(1, self.item_emb.num_embeddings, device=self.dev)
        return self.item_emb_module(all_item_ids)

    def log2feats(self, log_seqs):
        # 使用 item_emb_module 获取增强嵌入
        seqs = self.item_emb_module(log_seqs)
        # print(f"item_emb range: min={seqs.min().item()}, max={seqs.max().item()}, mean={seqs.mean().item()}")
        seqs *= self.item_emb.embedding_dim ** 0.5
        
        positions = torch.arange(1, log_seqs.shape[1] + 1, device=self.dev).unsqueeze(0).repeat(log_seqs.shape[0], 1)
        positions *= (log_seqs != 0)
        
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        timeline_mask = (log_seqs == 0)
        seqs *= ~timeline_mask.unsqueeze(-1)
        
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        # print(f"log_feats_pre range: min={seqs.min().item()}, max={seqs.max().item()}, mean={seqs.mean().item()}")

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q.transpose(0, 1), seqs.transpose(0, 1), seqs.transpose(0, 1),
                                                    attn_mask=attention_mask)
            # print(f"Q range: min={Q.min().item()}, max={Q.max().item()}, mean={Q.mean().item()}")
            # print(f"mha_outputs_{i} range: min={mha_outputs.min().item()}, max={mha_outputs.max().item()}, mean={mha_outputs.mean().item()}")
            # print(mha_outputs.shape)

            # # 检查 mha_outputs 中是否有 NaN
            # if torch.isnan(mha_outputs).any():
            #     # 找到所有 NaN 的坐标
            #     nan_positions = torch.nonzero(torch.isnan(mha_outputs), as_tuple=False)
            #     unique_positions = set()  # 用于存储唯一的前两位坐标
            #     print("mha_outputs 中的 NaN 坐标（只保留前两位，去重）:")
            #     for pos in nan_positions:
            #         # 只保留前两位坐标
            #         coord = tuple(pos.cpu().numpy()[:2])
            #         unique_positions.add(coord)  # 添加到集合中以去重
            #     sorted_positions = sorted(unique_positions)

            #     # 打印唯一的坐标
            #     for coord in sorted_positions:
            #         print(coord, log_seqs[coord[1], coord[0]].item(), timeline_mask[coord[1], coord[0]].item())
            # else:
            #     print("mha_outputs 中没有 NaN 值。")
            #     zero_positions = torch.nonzero(log_seqs == 0, as_tuple=False)
            #     # 输出结果
            #     print("所有 0 的二维坐标:")
            #     for pos in zero_positions:
            #         print(tuple(pos.cpu().numpy()))  # 将坐标移回 CPU 并转换为 NumPy 数组以便打印
            #     zero_positions = torch.nonzero(timeline_mask, as_tuple=False)
            #     # 输出结果
            #     print("所有 true 的二维坐标:")
            #     for pos in zero_positions:
            #         print(tuple(pos.cpu().numpy()))  # 将坐标移回 CPU 并转换为 NumPy 数组以便打印
            # mha_outputs[torch.isnan(mha_outputs)] = 0

            seqs = Q + mha_outputs.transpose(0, 1)
            # print(f"log_feats_{i}_1 range: min={seqs.min().item()}, max={seqs.max().item()}, mean={seqs.mean().item()}")
            seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))
            # print(f"log_feats_{i}_2 range: min={seqs.min().item()}, max={seqs.max().item()}, mean={seqs.mean().item()}")
            seqs *= ~timeline_mask.unsqueeze(-1)
            # print(f"log_feats_{i} range: min={seqs.min().item()}, max={seqs.max().item()}, mean={seqs.mean().item()}")

        return self.last_layernorm(seqs)

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_seqs_tensor = torch.LongTensor(log_seqs).to(self.dev)
        pos_seqs_tensor = torch.LongTensor(pos_seqs).to(self.dev)
        neg_seqs_tensor = torch.LongTensor(neg_seqs).to(self.dev)

        log_feats = self.log2feats(log_seqs_tensor)
        # print(f"log_feats range: min={log_feats.min().item()}, max={log_feats.max().item()}, mean={log_feats.mean().item()}")

        pos_embs = self.item_emb_module(pos_seqs_tensor)
        neg_embs = self.item_emb_module(neg_seqs_tensor)
        # print(f"pos_embs range: min={pos_embs.min().item()}, max={pos_embs.max().item()}, mean={pos_embs.mean().item()}")
        # print(f"neg_embs range: min={neg_embs.min().item()}, max={neg_embs.max().item()}, mean={neg_embs.mean().item()}")
        
        # print(f"log_seq shape:{log_seqs.shape}, pos_seq shape:{pos_seqs.shape}, neg_seq shape:{neg_seqs.shape}, log_feats shape:{log_feats.shape}, pos_embs shape:{pos_embs.shape}")
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        log_seqs_tensor = torch.LongTensor(log_seqs).to(self.dev)
        item_indices_tensor = torch.LongTensor(item_indices).to(self.dev)

        log_feats = self.log2feats(log_seqs_tensor)
        final_feat = log_feats[:, -1, :]

        item_embs = self.item_emb_module(item_indices_tensor)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits