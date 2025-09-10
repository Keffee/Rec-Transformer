# 含有feature处理的数据更新说明
## 关于文件结构和剩下的部分
剩下的部分完全对齐no feature部分（好吧其实是没改完，rq早停还没加上）

## 关于如何使用新版的3_data_to_embedding中的带特征sasrec
1. Data split by timesamp
```sh
python 1_1_generate_positive_train_test_based_on_timestamp.py
```
You will get `1_1_train.csv`和`1_1_test.csv`

2. 运行位于SASRec.pytorch文件夹中的`run.sh`就行，可能唯一要改的就是`--feature_path`和`--input_csv_path`的文件地址，同样地读取train文件和特征文件，输出的是checkpoint。

(运行这个文件：`Rec-Transformer/data/KuaiRand-27K-with_feature/34_from_data_to_rq_code/3_data_to_embedding/SASRec.pytorch/python/generate_npy.py`，也要对应改一下文件中的路径，传入参数直接复制`run.sh`中的就行，这一步是为了生成对应embedding的npy文件，因为这一步还是需要feature info的，所以args里面的`--feature_path`也得对应改了。这次输出的是npy文件。这一次的npy文件不再需要remap了，直接使用就行了，从第0行开始，第k行就是对应id=k的item.)

3. 关于4_embedding_to_rq_code
目前还没加早停之类的...其实也就是去掉了remap，我这边还没找到一个很好的方法来避免code冲突。
要运行的话还是
```sh
python our_train_and_generate.py
```

会在此目录下生成`rqvae_output`文件夹，并包含`original_item_id_to_rq_code.json`文件，此文件正是原本的正样例序列中的item_id到rq_code的映射，复制到本目录下重命名为`4_item_id_to_rq_code.json`.

4. 在本目录下运行`5_generate_rq_codes_pt_data.py`，它会利用`1_1_test.csv`和`4_item_id_to_rq_code.json`生成最终的符合LLaMA的pt格式的json`5_rq_codes_pt_data.json`，token之间用空格隔开，没出现在train中的id会自动补一个通用rq-code，padding会补成"0 0 0"

5. 在本目录下运行`6_data_transform_from_pt_json_2_train_test_parquet.py`，它会读取`1_1_test.csv`和`5_rq_codes_pt_data.json`在`6_parquet_for_verl`生成train和test的parquet，其中extra_info中已经包含user_id（此user_id是根据对应第几行从`1_1_test.csv`中获取的），然后按理来说会切分成answer部分是timestamp之后的子序列，input部分是timestamp之前的子序列，但是还是包含了padding，时间紧张所以没来得及优化。