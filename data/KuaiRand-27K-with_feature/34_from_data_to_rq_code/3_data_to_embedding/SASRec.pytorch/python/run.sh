python ./python/main.py \
  --dataset=KuaiRand_27K \
  --train_dir=default \
  --maxlen=10000 \
  --dropout_rate=0.2 \
  --batch_size=6 \
  --num_epochs=1000 \
  --feature_emb_dim=64 \
  --hidden_units=128 \
  --patience=3 \
  --device=cuda:0

  #--feature_path=/home/jovyan/tmp/kuairand/KuaiRand_27k/item_feat_norm.pkl \
  #--input_csv_path=/home/jovyan/Fuxi-OneRec/Rec-Transformer/data/KuaiRand-27K-no-feature/1_1_train.csv \