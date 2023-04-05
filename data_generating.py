
from arguments import args
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import add_self_loops


def compute_path_losses(args, distances):
    # f_min = args.carrier_f_start
    # f_max = args.carrier_f_end
    # f_gap = f_max-f_min
    # # carrier_lam = args.carrier_lam          # len N verctor
    # server_range = np.arange(server_num)
    # carrier_f = f_min + server_range * f_gap
    carrier_f = (args.carrier_f_start+args.carrier_f_end)/2
    carrier_lam = 2.998e8 / carrier_f
    # carrier_lam = np.concatenate([carrier_lam, np.zeros(args.max_server_num-server_num)], axis=0)
    signal_cof = args.signal_cof
    path_losses = (signal_cof * carrier_lam) / (distances**2)

    return path_losses

def generate_data(num_users):
    # 生成用户的任务大小和资源大小
    task_sizes = np.random.uniform(args.min_task_size, args.max_task_size, size=num_users)
    resource_sizes = np.random.uniform(args.min_compute_resource, args.max_compute_resource, size=num_users)
    max_powers = np.random.uniform(args.min_trans_power, args.max_trans_power, size=num_users)

    # 生成用户在环境中的随机位置
    positions = np.random.uniform(0, args.env_max_len, size=(num_users, 2))

    # 计算每对用户之间的欧几里得距离矩阵
    distance_matrix = np.sqrt(np.sum((positions[:, None] - positions) ** 2, axis=-1))

    # 创建一个图数据对象
    node_features = []

    # 遍历所有的节点，计算节点特征和边
    for i in range(num_users):
        # 创建节点特征向量
        node_feature = np.concatenate([np.array([task_sizes[i], resource_sizes[i], max_powers[i]]), np.ones(1)])
        node_features.append(node_feature)
    edge_src, edge_dst = np.where(distance_matrix>=args.dist_threshold)
    edge_index = np.concatenate([edge_src[np.newaxis, :], edge_dst[np.newaxis, :]], axis=0)
    edge_attr = distance_matrix[edge_src, edge_dst][:, np.newaxis]


    # 将节点特征和边特征转换为torch张量
    x = torch.tensor(node_features, dtype=torch.float, device=args.device)
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=args.device)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float, device=args.device)
    
    # 添加自环
    edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, fill_value=0)
    
    channel_gain = compute_path_losses(args, edge_attr)
    channel_gain[edge_index[0]==edge_index[1]] = 0

    # 创建图数据对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, channel_gain=channel_gain)
    return data


def generate_multiple_data(num_samples, batch_size):
    # 创建一个空列表，用于存储所有的数据样本
    data_list = []

    # 遍历所有的样本数量，生成相应数量的数据样本
    for num_users in num_samples:
        # 生成一个随机数量的用户
        # num_users = np.random.randint(min_num_users, max_num_users + 1)

        # 生成一组数据样本
        data = generate_data(num_users)

        # 将数据样本添加到列表中
        data_list.append(data)

    # 将所有数据样本打包成一个DataLoader对象，以便于进行batch加载
    loader = DataLoader(data_list, batch_size=batch_size)

    return loader
