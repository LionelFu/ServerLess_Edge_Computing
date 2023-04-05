import math
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor, Tensor
from torch_geometric.utils import softmax, remove_self_loops, contains_self_loops

class TransformerNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, head_num, edge_dim):
        super(TransformerNet, self).__init__()
        self.transformer_conv1 = TransformerConv(input_dim, hidden_dim, heads=head_num, edge_dim=edge_dim)
        self.transformer_conv2 = TransformerConv(hidden_dim * head_num, hidden_dim, heads=head_num, edge_dim=edge_dim)
        # self.transformer_conv3 = TransformerConv(128 * 8, 256, heads=8, edge_dim=edge_dim)
        # self.transformer_conv4 = TransformerConv(256 * 8, 512, heads=8, edge_dim=edge_dim)
        # self.linear = torch.nn.Linear(512 * 8, num_classes)

    def forward(self, x, edge_index, edge_attr):
        # x, batch = to_dense_batch(x, batch)
        # adj = to_dense_adj(edge_index, batch)
        # Compute PairTensor format.
        # src_idx, dst_idx = edge_index[0], edge_index[1]
        # edge_index = torch.stack([batch[src_idx], batch[dst_idx]], dim=0)
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.transformer_conv1(x, edge_index, edge_attr))
        x = F.relu(self.transformer_conv2(x, edge_index, edge_attr))
        # x = F.relu(self.transformer_conv3(x, edge_index, edge_attr))
        # x = F.relu(self.transformer_conv4(x, edge_index, edge_attr))
        # x = torch_geometric.nn.global_max_pool(x, batch)
        # x = self.linear(x)
        task_allocation_sche = self.transformer_conv2.task_allocation_sche
        power_allocation_sche = self.transformer_conv2.power_allocation_sche
        compute_allcoation_sche = self.transformer_conv2.compute_allocation_sche
        return task_allocation_sche, power_allocation_sche, compute_allcoation_sche



class TransformerConv(MessagePassing):

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.task_offloading_mlp = nn.Sequential(
                        nn.Linear(heads*out_channels, out_channels),
                        nn.ReLU(),
                        nn.Linear(out_channels, 1),
                        nn.LeakyReLU()
        )
        self.power_allocation_mlp = nn.Sequential(
                        nn.Linear(heads*out_channels, out_channels),
                        nn.ReLU(),
                        nn.Linear(out_channels, 1),
                        nn.LeakyReLU()
        )

        self.compute_allocation_mlp = nn.Sequential(
                        nn.Linear(heads*out_channels, out_channels),
                        nn.ReLU(),
                        nn.Linear(out_channels, 1),
                        nn.LeakyReLU()
        )

        self.reset_parameters()

    def reset_parameters(self):
        # super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        self.edge_index = edge_index
        # self.edge_index_wo_sloop, _ = remove_self_loops(self.edge_index)
        self.local_self_loops = contains_self_loops(edge_index)
        self.wo_self_loops_mask = torch.zeros((edge_index.shape[1]), device=x.device)
        self.wo_self_loops_mask[torch.where(edge_index[0]==edge_index[1])[0]] = -torch.inf
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        task_allocation_sche = self.task_offloading_mlp(out.view(-1, self.heads*self.out_channels))
        power_allocation_sche = self.power_allocation_mlp(out.view(-1, self.heads*self.out_channels))
        power_allocation_sche = power_allocation_sche + self.wo_self_loops_mask.unsqueeze(-1) # 没有自环的设为0

        compute_allocation_sche = self.compute_allocation_mlp(out.view(-1, self.heads*self.out_channels))

        self.task_allocation_sche = softmax(task_allocation_sche, index=self.edge_index[0])
        self.power_allocation_sche = softmax(power_allocation_sche, index=self.edge_index[0])
        self.compute_allocation_sche = softmax(compute_allocation_sche, index=self.edge_index[1])
        
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')