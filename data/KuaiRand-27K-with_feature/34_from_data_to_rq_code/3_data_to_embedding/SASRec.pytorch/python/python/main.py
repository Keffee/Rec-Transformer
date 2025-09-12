import os
import time
import torch
# torch.autograd.set_detect_anomaly(True)
import argparse

from model import SASRec
from utils import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.nn as nn
def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True,choices=["KuaiRand-27K", "KuaiRand-27K-0501","KuaiRand-27K-100krows"])
#parser.add_argument('--train_dir', required=True,help='The dataset variant name')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--norm_first', action='store_true', default=False)
parser.add_argument('--patience', default=5, type=int, help='Number of epochs to wait for improvement before early stopping.')
parser.add_argument('--feature_path', default=r'/zhdd/home/kfwang/20250813Reproduct_Onerec/Fuxi-OneRec/Rec-Transformer/data/KuaiRand-27K/KuaiRand-27K-Processed/item_feat_norm.pkl', type=str, help='Path to the item features .pkl file.')
parser.add_argument('--input_csv_path', default=r'/zhdd/home/kfwang/20250813Reproduct_Onerec/Fuxi-OneRec/Rec-Transformer/data/KuaiRand-27K/KuaiRand-27K-Processed/1_1_KuaiRand-27K.csv', type=str, help='Path to the train csv file.')
parser.add_argument('--feature_emb_dim', default=5, type=int,help='The dim for context feature.')

args = parser.parse_args()

base_output_dir = "../../../../output_"+args.dataset

args.feature_path = os.path.join(base_output_dir, "item_feat_norm.pkl")
args.input_csv_path = os.path.join(base_output_dir, "1_1_train.csv")

# Define finetuned parameters and show them in output folder name. 
# If already select the best parameter, just remove it.
run_parts = [
    f"bs{args.batch_size}",
    f"lr{str(args.lr)}",
    f"L{args.maxlen}",
    f"d{args.hidden_units}",
    f"blk{args.num_blocks}",
    f"h{args.num_heads}",
    f"fd{args.feature_emb_dim}",
]

# all output files will be saved into run_dir("KuaiRand_27K_default/bs_xxx/"). The datetime is optional. 
run_name = "_".join(run_parts) #+ "_" + datetime.now().strftime("%Y%m%d")
run_dir = os.path.join(f"output_{args.dataset}", run_name)
os.makedirs(run_dir, exist_ok=True)
best_model_path = os.path.join(run_dir, "SASRec_best.pth")
# Save args
with open(os.path.join(run_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([f"{k},{v}" for k, v in sorted(vars(args).items())]))

# tensorboard folder
tblog_dir = os.path.join(run_dir, 'tblog')
if not os.path.exists(tblog_dir):
    os.makedirs(tblog_dir)    
writer = SummaryWriter(log_dir=tblog_dir)

if __name__ == '__main__':
    # # global dataset
    # # dataset = data_partition(args.dataset)
    # csv_file_path = r'/home/kfwang/20250813复现Onerec/Fuxi-OneRec/Rec-Transformer/data/KuaiRand-27K/KuaiRand-27K-Processed/1_positive_data_0.csv'
    # u2i_index, i2u_index = build_index_from_csv(csv_file_path)
    # dataset = csv_data_partition(csv_file_path)

    # 1. 定义你的数据文件所在的目录和文件命名规则
    #input_csv_path = r'/zhdd/home/kfwang/20250813Reproduct_Onerec/Fuxi-OneRec/Rec-Transformer/data/KuaiRand-27K/KuaiRand-27K-Processed/1_1_train.csv'
    # # 文件名的模板，{}是后续用来填充数字的占位符
    # filename_pattern = "1_positive_data_{}.csv" 
    # # 你想要加载的文件的编号，range(4) 会生成 0, 1, 2, 3
    # file_indices_to_load = range(4) 

    # 2. 调用新函数来加载并合并指定的数据文件
    combined_data_df = load_single_file_as_dataframe(args.input_csv_path)

    # 3. 检查数据是否成功加载，然后传递给处理函数
    if combined_data_df is not None:
        dataset = partition_data_from_dataframe(combined_data_df)
        [user_train, user_valid, user_test, usernum, itemnum] = dataset
        
        # --- 修改：调用新的特征处理函数 ---
        feature_info = None
        if args.feature_path:
            feature_info = load_and_preprocess_features(args.feature_path, itemnum)
            if feature_info is None:
                print("特征加载失败，将不使用特征进行训练。")
        else:
            print("未提供特征文件路径，将不使用特征进行训练。")
        # --- 修改结束 ---
    else:
        raise ValueError("combined_data_df is None")

    # num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    seqlen = []
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
        seqlen.append(len(user_train[u]))
    seqlen = np.array(seqlen)

    # 基本统计
    min_len = seqlen.min()
    max_len = seqlen.max()
    q1 = np.percentile(seqlen, 25)  # 第一四分位
    median = np.percentile(seqlen, 50)  # 中位数
    q3 = np.percentile(seqlen, 75)  # 第三四分位
    mean_len = seqlen.mean()
    q95 = np.percentile(seqlen, 95)

    print(f"min={min_len}, max={max_len}")
    print(f"Q1={q1}, median={median}, Q3={q3}, Q95={q95}")
    print(f"mean={mean_len:.2f}")

    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    #f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    f = open(os.path.join(run_dir, 'log.txt'), 'w')
    #f.write('epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n')
    f.write('epoch (val_ndcg, val_hr)\n')
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(usernum, itemnum, args, feature_info=feature_info).to(args.device) # no ReLU activation in original SASRec implementation?
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)

    def count_dense_sparse_in_m(model):
        total, dense, sparse = 0, 0, 0
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                for name, param in module.named_parameters(recurse=False):
                    sparse += param.numel()
            else:
                for name, param in module.named_parameters(recurse=False):
                    dense += param.numel()
        total = dense + sparse

        print(f"Total params : {total/1e6:.3f} M")
        print(f"Dense params : {dense/1e6:.3f} M")
        print(f"Sparse params: {sparse/1e6:.3f} M")

        return total, dense, sparse

    count_dense_sparse_in_m(model)

    model.train() # enable model training
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
            
    
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
    
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
        
    patience = args.patience
    early_stop_counter = 0
    # 我们将使用验证集上的 NDCG@10 作为早停的监控指标
    best_metric_for_early_stop = 0 
    # =================================================================
    running_loss, global_step = 0.0, 1 
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            model.train()
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)

            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            # print("\neye ball check raw_logits:"); print(pos_logits[indices]); print(neg_logits[indices]) # check pos_logits > 0, neg_logits < 0
            # print("\neye ball check raw_lables:"); print(pos_labels[indices]); print(neg_labels[indices]) # check pos_logits > 0, neg_logits < 0
            # # 诊断代码：检查 logits 的范围
            # print(f"pos_logits range: min={pos_logits.min().item()}, max={pos_logits.max().item()}, mean={pos_logits.mean().item()}")
            # print(f"neg_logits range: min={neg_logits.min().item()}, max={neg_logits.max().item()}, mean={neg_logits.mean().item()}")

            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # max_norm 是超参数，可以从 1.0 或 5.0 开始尝试
            adam_optimizer.step()
            # 移除这行打印可以加速训练
            # print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs
            running_loss += loss.item()
            avg_loss = running_loss / global_step
            writer.add_scalar("Loss/Avg_per_step", avg_loss, global_step)
            global_step+=1
            
        if epoch % 2 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            eval_st = time.time()
            #t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            #print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
            #        % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1]))            
            eval_et = time.time()-eval_st
            print(f"eval time: {eval_et:.2f}s")
            writer.add_scalar("Eval/NDCG@10_per_2epoch", t_valid[0], epoch)
            writer.add_scalar("Eval/HR@10_per_2epoch", t_valid[1], epoch)
            #writer.add_scalar("Test/NDCG@10_per_2epoch", t_test[0], epoch)
            #writer.add_scalar("Test/HR@10_per_2epoch", t_test[1], epoch)

            # 原有的保存最佳模型的逻辑（当任何一个指标提升时）
            if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr: #or t_test[0] > best_test_ndcg or t_test[1] > best_test_hr:
                best_val_ndcg = max(t_valid[0], best_val_ndcg)
                best_val_hr = max(t_valid[1], best_val_hr)
                #best_test_ndcg = max(t_test[0], best_test_ndcg)
                #best_test_hr = max(t_test[1], best_test_hr)
                
                #folder = args.dataset + '_' + args.train_dir
                #fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                #fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                #best_model_path = os.path.join(folder, fname)
                #print(f"New best model found. Saving to {best_model_path}")
                #torch.save(model.state_dict(), os.path.join(folder, fname))
                # only save one best model into run_dir
                
                print(f"New best model found. Saving to {run_dir}")
                torch.save(model.state_dict(), best_model_path)
            # =================================================================
            # 新增：早停判断逻辑
            current_metric = t_valid[0] # 使用验证集 NDCG@10
            if current_metric > best_metric_for_early_stop:
                best_metric_for_early_stop = current_metric
                early_stop_counter = 0
                print("Validation metric improved, resetting early stopping counter.")
            else:
                early_stop_counter += 1
                print(f"Validation metric did not improve. Early stopping counter: {early_stop_counter}/{patience}")

            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {patience} evaluations without improvement.")
                break # 跳出主训练循环
            # =================================================================

            #f.write(str(epoch) + ' ' + str(t_valid) + ' ' + str(t_test) + '\n')
            f.write(str(epoch) + ' ' + str(t_valid) + '\n')
            f.flush()
            t0 = time.time()
            model.train()
    
        if epoch == args.num_epochs:
            #folder = args.dataset + '_' + args.train_dir
            #fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            #fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            #torch.save(model.state_dict(), os.path.join(folder, fname))
            torch.save(model.state_dict(), os.path.join(run_dir, "SASRec_best.pth"))
    print("Training finished.") # 新增：告知训练结束
    f.close()
    sampler.close()

    # --- MODIFICATION 3: 在所有流程结束后，调用嵌入提取函数 ---
    if 'best_model_path' in locals() and best_model_path:
        #output_folder = args.dataset + '_' + args.train_dir
        #embedding_output_file = os.path.join(output_folder, 'best_item_embeddings.npy')
        #embedding_output_file = os.path.join(run_dir, 'best_item_embeddings.npy')
        embedding_output_file = os.path.join(base_output_dir, 'best_item_embeddings.npy')
        # 重新加载最佳模型权重到当前模型结构
        print(f"\n重新加载最佳模型权重从: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=torch.device(args.device)))

        #extract_and_save_item_embeddings(
        #    model=model,
        #    output_path=embedding_output_file
        #)
        extract_and_save_item_embeddings_batch(
            model=model,
            output_path=embedding_output_file
        )        
    else:
        print("\n训练期间没有保存任何最佳模型，跳过嵌入提取步骤。")
    print("Done")