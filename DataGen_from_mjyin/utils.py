import torch

from datasets import Dataset

default_dataset_config = {
    'ml-1m': {
        'dataset_path': "data/ml-1m/ratings.jsonl",
        'user_column': 'user_id',
        'item_column': 'item_id',
        'timestamp_column': 'timestamp',
        'k_core': 0,
    },
    'amazon-beauty': {
        'dataset_path': "data/amazon-beauty/All_Beauty.jsonl",
        'user_column': 'user_id',
        'item_column': 'parent_asin',
        'timestamp_column': 'timestamp',
        'k_core': 5,
    },
    'amazon-book': {
        'dataset_path': "data/amazon-book/Books.jsonl",
        'user_column': 'user_id',
        'item_column': 'parent_asin',
        'timestamp_column': 'timestamp',
        'k_core': 5,
    },
    'amazon-software': {
        'dataset_path': "data/amazon-software/Software.jsonl",
        'user_column': 'user_id',
        'item_column': 'parent_asin',
        'timestamp_column': 'timestamp',
        'k_core': 5,
    },
    'amazon-industrial': {
        'dataset_path': "data/amazon-industrial/Industrial_and_Scientific.jsonl",
        'user_column': 'user_id',
        'item_column': 'parent_asin',
        'timestamp_column': 'timestamp',
        'k_core': 5,
    },
}

def drop_last_n_items(sample, drop_last_n):
    for k, v in sample.items():
        if k.endswith('sequence'):
            assert drop_last_n != 0
            if type(v) is list:
                sample[k] = v[:-drop_last_n]
            elif type(v) is str:
                sample[k] = ' '.join(v.split()[:-drop_last_n])
    return sample

def update_config_by_default_data_config(config):
    dataset_name = config.dataset_name
    default_config = default_dataset_config[dataset_name]
    config.dataset_path = default_config['dataset_path'] if config.dataset_path is None else config.dataset_path
    config.user_column = default_config['user_column'] if config.user_column is None else config.user_column
    config.item_column = default_config['item_column'] if config.item_column is None else config.item_column
    config.timestamp_column = default_config['timestamp_column'] if config.timestamp_column is None else config.timestamp_column
    config.k_core = default_config['k_core'] if config.k_core is None else config.k_core
    return config

def to_int_list(sample):
    sample['item_id_list:token_seq'] = [int(_) for _ in sample['item_id_list:token_seq'].split()]
    return sample

def condense_seq_data(dataset : Dataset, max_seq_len):
    dataset = dataset.map(to_int_list, num_proc=32)
    user_id, item_id_list, item_id = dataset['user_id:token'], dataset['item_id_list:token_seq'], dataset['item_id:token']
    item_id_list = [_user_seq + [_target_item] for (_user_seq, _target_item) in zip(item_id_list, item_id)]
    seq_len = torch.tensor([len(_) for _ in item_id_list])
    sorted_seq_len, sorted_index = torch.sort(seq_len, descending=True)
    sorted_seq_len = sorted_seq_len.tolist()
    sorted_item_id_list = [item_id_list[idx] for idx in sorted_index]
    # sorted_item_id = [item_id[idx] for idx in sorted_index]
    # sorted_item_id_list = item_id_list[sorted_index]
    # sorted_item_id = item_id[sorted_index].tolist()
    # sorted_item_id_list = [_user_seq + [_target_item] for (_user_seq, _target_item) in zip(sorted_item_id_list, sorted_item_id)]
    merged_data = {'user_id:token': [], 'item_id_list:token_seq': [], 'item_id:token': []}

    pre_pointer, post_pointer = 0, len(item_id_list) - 1
    new_data_cnt = 0
    while(pre_pointer <= post_pointer):
        cur_seq, cur_seq_len = sorted_item_id_list[pre_pointer], sorted_seq_len[pre_pointer]
        while cur_seq_len <= max_seq_len:
            post_seq_len = sorted_seq_len[post_pointer]
            if (cur_seq_len + post_seq_len <= max_seq_len) and pre_pointer != post_pointer:
                cur_seq = cur_seq[:cur_seq_len] + sorted_item_id_list[post_pointer][:post_seq_len]
                cur_seq_len = cur_seq_len + post_seq_len
                post_pointer -= 1
            else:# Record this sequence
                cur_seq = cur_seq[:cur_seq_len]
                merged_data['user_id:token'].append(new_data_cnt)
                merged_data['item_id_list:token_seq'].append(' '.join([str(_) for _ in cur_seq[:-1]]))
                merged_data['item_id:token'].append(cur_seq[-1])
                pre_pointer += 1
                new_data_cnt += 1
                break
    
    return Dataset.from_dict(merged_data)