处理数据的流程有点长，先把pipeline记下来

1. 利用本目录下的`1_1_generate_positive_train_test_based_on_timestamp.py`,生成正样例序列`1_1_train.csv`和`1_1_test.csv`.

2. 利用本目录下的`2_build_remaped_ids.py`，将正样例序列压缩成**从1开始**的无空隙正样例序列`2_remapped_positive_data_100k.csv`，便于后续SASRec操作，并且顺便存下来压缩前后user_id和item_id到新id的映射`2_user_id_map.csv`, `2_item_id_map.csv`.

3. 运行文件夹`34_from_data_to_rq_code/3_data_to_embedding/SASRec.pytorch`中的`run.sh`，命令行参数为
会在`34_from_data_to_rq_code/3_data_to_embedding/SASRec.pytorch/KuaiRand_27K_default`中生成最优模型中每个item的embedding`best_item_embeddings.npy`，其中0号位置是padding，不用管，复制到本目录下重命名为`3_IdZeroIsPadding_item_embeddings.npy`

4. 直接在目录`34_from_data_to_rq_code/4_embedidng_to_rq_code`下运行
```sh
python our_train_and_generate.py
```
会在此目录下生成`rqvae_output`文件夹，并包含`original_item_id_to_rq_code.json`文件，此文件正是原本的正样例序列中的item_id到rq_code的映射，复制到本目录下重命名为`4_item_id_to_rq_code.json`.

5. 在本目录下运行`5_generate_rq_codes_pt_data.py`，它会利用`1_1_test.csv`和`4_item_id_to_rq_code.json`生成最终的符合LLaMA的pt格式的json`5_rq_codes_pt_data.json`，token之间用空格隔开，没出现在train中的id会自动补一个通用rq-code，padding会补成"0 0 0"

6. 在本目录下运行`6_data_transform_from_pt_json_2_train_test_parquet.py`，它会读取`1_1_test.csv`和`5_rq_codes_pt_data.json`在`6_parquet_for_verl`生成train和test的parquet，其中extra_info中已经包含user_id（此user_id是根据对应第几行从`1_1_test.csv`中获取的），然后按理来说会切分成answer部分是timestamp之后的子序列，input部分是timestamp之前的子序列，但是还是包含了padding，时间紧张所以没来得及优化。