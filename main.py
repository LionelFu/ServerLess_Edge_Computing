import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from models import TransformerNet
from data_generating import generate_multiple_data
from arguments import args
import numpy as np
import random

def compute_time_loss(power_allocation, task_allocation, compute_allocation, src_idx, tgt_idx, task_size, max_power, max_compute_resource, channel_gain):
    
    '''
        power_allocation: 自环的功率分配为0
        task_allocation: 自环和其他边一致
        compute_resource: 
    '''
    
    eps = 1e-20   # 极小值

    n = src_idx[-1]+1# 用户个数

    max_compute_res = max_compute_resource[tgt_idx].unsqueeze(-1)
    compute_res = max_compute_res * compute_allocation
    # 对于compute_res分配过少的用户，不对其分配task, 将task分摊给其他邻接节点进行
    kik_idx = torch.where(compute_res < args.lower_bound_compute_resource)[0]
    task_allocation_clone = task_allocation.clone()
    task_allocation_clone[kik_idx] = 0
    task_allo = torch.zeros(n, device=args.device)
    task_allo.index_add_(0, src_idx, task_allocation_clone.squeeze())
    # 处理没有用户给其分配计算资源的情况：全部交给自环
    non_allocated_idx = torch.where(task_allo==0)
    task_allo[non_allocated_idx] = 1
    task_allocation_clone[src_idx==non_allocated_idx and tgt_idx==non_allocated_idx] = 1
    task_allo = task_allo[src_idx]
    task_allocation_clone = torch.div(task_allocation_clone, task_allo.unsqueeze(-1)+eps)

    max_power = max_power[src_idx].unsqueeze(-1)
    pw = power_allocation * max_power * channel_gain
    # src_pw = torch.zeros(n)
    tgt_pw = torch.zeros(n, device=args.device)
    # src_pw.index_add_(0, src_idx, pw)
    tgt_pw.index_add_(0, tgt_idx, pw.squeeze())   # 每个用户接收的总pw
    
    # 计算传输时间
    # interference = interference_matrix.sum(dim=0)
    interference = tgt_pw[tgt_idx].unsqueeze(-1)
    snr = torch.div(pw, (interference + 1))
    rate = args.bandwidth * torch.log2(1+snr)
    tasks = task_size[src_idx].unsqueeze(-1) * task_allocation_clone
    
    # rate==0时传输时间也应为0
    tasks_clone = tasks.clone()
    tasks_clone[rate==0] = 0
    trans_time = torch.div(tasks_clone, rate+eps)     # 每条边的传输时间
    
    # 计算每个用户的总传输时间
    # user_trans_time = torch.zeros(n, device=args.device)    # 每个用户的传输时间
    # user_trans_time.index_add_(0, src_idx, trans_time)
    
    compute_time = torch.div(tasks, compute_res+eps)
    # user_compute_time = torch.zeros(n, device=args.device)
    # user_compute_time.index_add_(0, src_idx, compute_time)
    
    total_time = trans_time + compute_time
    user_total_time = torch.zeros(n, device=args.device)
    user_total_time.index_add_(0, src_idx, total_time.squeeze())

    return user_total_time.mean()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     np.random.seed(seed)

setup_seed(50)


Epochs = 20


# 生成示例数据
train_user_nums = np.random.randint(args.min_user_num, args.max_user_num, (args.train_layouts))
train_loader = generate_multiple_data(train_user_nums, args.batch_size)

# 调用模型进行处理
model = TransformerNet(args.input_dim, args.hidden_dim, args.head_num, args.edge_dim).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
for epoch in range(Epochs):
    loss = 0
    length = 0
    for data in train_loader:
        src_idx = data.edge_index[0]
        tgt_idx = data.edge_index[1]
        task_size = data.x[:, 0]
        max_power = data.x[:, 2]
        max_compute_resource = data.x[:, 1]
        channel_gain = data.channel_gain

        task_allocation_sche, power_allocation_sche, compute_allocation_sche = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
        average_time = compute_time_loss(power_allocation_sche, task_allocation_sche, compute_allocation_sche, src_idx, tgt_idx, task_size, max_power, max_compute_resource, channel_gain)
        average_time.backward()
        optimizer.step()
        loss += average_time.item()
        length += 1
    print('train loss: {}'.format(loss/length))




