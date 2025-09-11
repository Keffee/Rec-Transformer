# 含有feature处理的数据更新说明
## 关于文件结构和剩下的部分
剩下的部分完全对齐no feature部分（好吧其实是没改完，rq早停还没加上）

# 0. Structure

```sh
cd ~/Fuxi-OneRec/Rec-Transformer/data/KuaiRand-27K-with_feature
```

The structure in this folder is like this:

```bash
.KuaiRand-27K-with_feature
|____34_from_data_to_rq_code
| |____3_data_to_embedding
| |____4_embedidng_to_rq_code
|____readme.md
|____1_1_generate_positive_train_test_based_on_timestamp.py
|____5_generate_rq_codes_pt_data.py
|____6_data_transform_from_pt_json_2_train_test_parquet.py
|____output_KuaiRand-27K-0501
| |____1_1_train.csv
| |____1_1_test.csv
| |____dnn_feature_columns.pkl
| |____item_feat_norm.pkl
| |____user_feat_norm.pkl
```

All the output will be saved under folder `output_{dataset}`. dataset can be `["KuaiRand-27K", "KuaiRand-27K-0501","KuaiRand-27K-100krows"]`

# 1. Data split by timesamp
```sh
# for full data: 
python 1_1_generate_positive_train_test_based_on_timestamp.py --dataset KuaiRand-27K

# for small data"
python 1_1_generate_positive_train_test_based_on_timestamp.py --dataset KuaiRand-27K-0501
```

You will get `1_1_train.csv`和`1_1_test.csv` in `f"output_{args.dataset}"`

# 2. Run SASRec to get item embedding
```sh
cd 34_from_data_to_rq_code/3_data_to_embedding/SASRec.pytorch/python

DATASET="KuaiRand-27K-0501"
LEN=6000
BS=6
DIM_FEAT=64
DIM_HIDDEN=128
LR=0.001
PATIENCE=3

LOG_DIR="output_${DATASET}"
mkdir -p $LOG_DIR

LOG_FILE="${LOG_DIR}/train_len${LEN}_bs${BS}_feat${DIM_FEAT}_hidden${DIM_HIDDEN}_lr${LR}_pat${PATIENCE}.log"

CUDA_VISIBLE_DEVICES=1 nohup python ./python/main.py \
    --dataset=$DATASET \
    --maxlen=$LEN \
    --batch_size=$BS \
    --feature_emb_dim=$DIM_FEAT \
    --hidden_units=$DIM_HIDDEN \
    --patience=$PATIENCE \
    --lr=$LR \
    > $LOG_FILE 2>&1 &
```

You will get `best_item_embeddings.npy` in `f"output_{args.dataset}` 从第0行开始，第k行就是对应id=k的item.

# 3. 关于4_embedding_to_rq_code
还没找到一个很好的方法来避免code冲突。

```sh
DATASET="KuaiRand-27K-0501"
BS=4096
D=128
LR=1e-3
NUM_EMB_LIST="8000 8000 8000"

CUDA_VISIBLE_DEVICES=4 python our_train_and_generate.py \
  --dataset $DATASET \
  --e_dim $D \
  --lr $LR \
  --batch_size $BS \
  --num_emb_list $NUM_EMB_LIST \
  > output_KuaiRand-27K-0501/log_${DATASET}_bs${BS}_d${D}_lr${LR}_emb${NUM_EMB_LIST// /-}.out 2>&1 &
```

会在此目录下生成`rqvae_output`文件夹，并包含`original_item_id_to_rq_code.json`文件，此文件正是原本的正样例序列中的item_id到rq_code的映射，choose the best one 复制到`output_KuaiRand-27K-0501`目录下

4. 在本目录下运行

```sh
cd ~/Fuxi-OneRec/Rec-Transformer/data/KuaiRand-27K-with_feature/

python 5_generate_rq_codes_pt_data.py --dataset KuaiRand-27K-0501

```
它会利用`1_1_test.csv`和`4_item_id_to_rq_code.json`生成最终的符合LLaMA的pt格式的json`5_rq_codes_pt_data.json`，token之间用空格隔开，没出现在train中的id会自动补一个通用rq-code，padding会补成"0 0 0"

5. 在本目录下运行`6_data_transform_from_pt_json_2_train_test_parquet.py`，它会读取`1_1_test.csv`和`5_rq_codes_pt_data.json`在`6_parquet_for_verl`生成train和test的parquet，其中extra_info中已经包含user_id（此user_id是根据对应第几行从`1_1_test.csv`中获取的），然后按理来说会切分成answer部分是timestamp之后的子序列，input部分是timestamp之前的子序列，但是还是包含了padding，时间紧张所以没来得及优化。