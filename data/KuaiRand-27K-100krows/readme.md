处理数据的流程有点长，先把pipeline记下来

1. 利用本目录下的`1_select_positive_interactions.py`,生成正样例序列`1_positive_data_100k.csv`.

2. 利用本目录下的`2_build_remaped_ids.py`，将正样例序列压缩成**从1开始**的无空隙正样例序列`2_remapped_positive_data_100k.csv`，便于后续SASRec操作，并且顺便存下来压缩前后user_id和item_id到新id的映射`2_user_id_map.csv`, `2_item_id_map.csv`.

3. 运行文件夹`1_from_data_to_rq_code/data_to_embedding/SASRec.pytorch/python`中的`main.py`，命令行参数为
```sh
python main.py --dataset=KuaiRand_27K_100krows --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda
```
会在`1_from_data_to_rq_code/data_to_embedding/SASRec.pytorch/python/KuaiRand_27K_100krows_default`中生成最优模型中每个item的embedding`best_item_embeddings.npy`，其中0号位置是padding，不用管，复制到本目录下重命名为`3_IdZeroIsPadding_item_embeddings.npy`

4. 直接在目录`1_from_data_to_rq_code/embedidng_to_rq_code`下运行
```sh
python our_train_and_generate.py
```
会在此目录下生成`rqvae_output`文件夹，并包含`original_item_id_to_rq_code.json`文件，此文件正是原本的正样例序列中的item_id到rq_code的映射，复制到本目录下重命名为`4_item_id_to_rq_code.json`.

5. 在本目录下运行`5_generate_rq_codes_pt_data.py`，它会利用`1_positive_data_100k.csv`和`4_item_id_to_rq_code.json`生成最终的符合LLaMA的pt格式的json`5_rq_codes_pt_data.json`，token之间用空格隔开。

6. 在本目录下运行`6_data_transform_from_pt_json_2_train_test_parquet.py`，它会读取`1_positive_data_100k.csv`和`5_rq_codes_pt_data.json`在`6_parquet_for_verl`生成train和test的parquet，其中extra_info中已经包含user_id（此user_id是根据对应第几行从`1_positive_data_100k.csv`中获取的）