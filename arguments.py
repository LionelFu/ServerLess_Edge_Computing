import argparse

args = argparse.ArgumentParser()

# 环境参数
args.add_argument('--MEC_num', default=10)
args.add_argument('--dist_threshold', default=100)
args.add_argument('--env_max_len', default=400)
args.add_argument('--min_task_size', default=500)
args.add_argument('--max_task_size', default=1000)
args.add_argument('--min_compute_resource', default=500)
args.add_argument('--max_compute_resource', default=1000)
args.add_argument('--min_trans_power', default=0.5)
args.add_argument('--max_trans_power', default=1)
args.add_argument('--carrier_f_start', default=2.4e9)
args.add_argument('--carrier_f_end', default=2.4835e9)
args.add_argument('--signal_cof', default=4.11)
args.add_argument('--bandwidth', default=1e20)

# 模型参数
args.add_argument('--input_dim', default=4)
args.add_argument('--hidden_dim', default=32)
args.add_argument('--head_num', default=2)
args.add_argument('--edge_dim', default=1)


# 实验参数
args.add_argument('--device', default='cuda')
args.add_argument('--min_user_num', default=10)
args.add_argument('--max_user_num', default=15)
args.add_argument('--train_layouts', default=4096)
args.add_argument('--eval_layouts', default=64)
args.add_argument('--batch_size', default=32)
args.add_argument('--lr', default=1e-4)


args = args.parse_args()
print(args)

