#-*- encoding=utf-8 -*-
import torch
import torch.nn as nn
from utils.logging import get_logger
logger = get_logger()
"""
https://github.com/renjunxiang/Multihead-Attention/blob/master/Attention.py
"""
class Attention1(nn.Module):
    """
    1.输入 [N,T,C] -> Linear、Tanh
    2. -> [N,T,1] -> unsqueeze
    3. -> [N,T] -> Softmax
    4. -> [N,T] -> unsqueeze
    5. -> [N,1,T] -> repeat
    6. -> [N,C,T] -> transpose
    7. -> [N,T,C]
    """

    def __init__(self, hidden_dim):
        super(Attention1, self).__init__()
        self.hidden_dim = hidden_dim
        self.dense = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        batch_size, time_step, hidden_dim = features.size()
        weight = nn.Tanh()(self.dense(features)).squeeze(-1)


        mask_idx = torch.sign(torch.abs(features).sum(dim=-1))
        paddings = torch.ones_like(mask_idx) * (-2 ** 32 + 1)
        weight = torch.where(torch.eq(mask_idx, 1), weight, paddings)

        weight = nn.Softmax(dim=1)(weight)
        weight = weight.unsqueeze(1)
        weight = weight.repeat(1, hidden_dim, 1)
        weight = weight.transpose(2, 1)
        features_attention = weight * features

        return features_attention


class Attention2(nn.Module):
    """
    1.输入 [N,T,C] -> Linear、Tanh
    2. -> [N,T,C] -> transpose
    3. -> [N,C,T] -> Softmax
    4. -> [N,C,T] -> mean
    5. -> [N,T] -> unsqueeze
    5. -> [N,1,T] -> expand
    6. -> [N,C,T] -> transpose
    7. -> [N,T,C]
    """

    def __init__(self, hidden_dim):
        super(Attention2, self).__init__()
        self.hidden_dim = hidden_dim
        self.dense = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, features, mean=True):
        batch_size, time_step, hidden_dim = features.size()
        weight = nn.Tanh()(self.dense(features))

        # mask给负无穷使得权重为0
        mask_idx = torch.sign(torch.abs(features).sum(dim=-1))
        mask_idx = mask_idx.unsqueeze(-1).expand(batch_size, time_step, hidden_dim)
        paddings = torch.ones_like(mask_idx) * (-2 ** 32 + 1)
        weight = torch.where(torch.eq(mask_idx, 1), weight, paddings)

        weight = weight.transpose(2, 1)
        weight = nn.Softmax(dim=2)(weight)
        if mean:
            weight = weight.mean(dim=1)
            weight = weight.unsqueeze(1)
            weight = weight.repeat(1, hidden_dim, 1)
        weight = weight.transpose(2, 1)
        features_attention = weight * features

        return features_attention


class LayerNorm(nn.Module):


    def __init__(self, features, epsilon=1e-8):
        super(LayerNorm, self).__init__()
        self.beta = nn.Parameter(torch.zeros(features))
        self.gamma = nn.Parameter(torch.ones(features))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        normalized = (x - mean) / (std + self.epsilon)
        outputs = self.gamma * normalized + self.beta

        return outputs


class Multihead_Attention(nn.Module):
    """
    multihead_attention
    1.split+cat
    2.matmul(q,k)
    3.mask k
    4.softmax
    5.mask q
    6.matmul(attn,v)
    7.split+cat
    8.res q
    9.norm
    """

    def __init__(self,
                 hidden_dim,
                 C_q=None,
                 C_k=None,
                 C_v=None,
                 num_heads=1,
                 dropout_rate=0.0):
        super(Multihead_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        C_q = C_q if C_q else hidden_dim
        C_k = C_k if C_k else hidden_dim
        C_v = C_v if C_v else hidden_dim
        self.linear_Q = nn.Linear(C_q, hidden_dim)
        self.linear_K = nn.Linear(C_k, hidden_dim)
        # self.linear_V = nn.Linear(C_v, hidden_dim)
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self,
                Q, K, V):
        """
        :param Q: A 3d tensor with shape of [N, T_q, C_q]
        :param K: A 3d tensor with shape of [N, T_k, C_k]
        :param V: A 3d tensor with shape of [N, T_v, C_v]
        :return:
        """
        num_heads = self.num_heads
        N = Q.size()[0]

        # Linear projections
        Q_l = nn.ReLU()(self.linear_Q(Q))
        K_l = nn.ReLU()(self.linear_K(K))
        # V_l = V  #do nothing to V
        # V_l = nn.ReLU()(self.linear_V(V))

        # Split and concat
        Q_split = Q_l.split(split_size=self.hidden_dim // num_heads, dim=2)
        K_split = K_l.split(split_size=self.hidden_dim // num_heads, dim=2)
        # V_split = V_l.split(split_size=self.hidden_dim // num_heads, dim=2)

        Q_ = torch.cat(Q_split, dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(K_split, dim=0)  # (h*N, T_k, C/h)
        # V_ = torch.cat(V_split, dim=0)  # (h*N, T_v, C/h)
        V_ = V.repeat(num_heads, 1, 1)

        # Multiplication
        outputs = torch.bmm(Q_, K_.transpose(2, 1))

        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)

        # Key Masking
        key_masks = torch.sign(torch.abs(K).sum(dim=-1))  # (N, T_k)
        key_masks = key_masks.repeat(num_heads, 1)  # (h*N, T_k)
        key_masks = key_masks.unsqueeze(1).repeat(1, Q.size()[1], 1)  # (h*N, T_q, T_k)

        paddings = torch.ones_like(key_masks) * (-2 ** 32 + 1)
        outputs = torch.where(torch.eq(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = nn.Softmax(dim=2)(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = torch.sign(torch.abs(Q).sum(dim=-1))  # (N, T_q)
        query_masks = query_masks.repeat(num_heads, 1)  # (h*N, T_q)
        query_masks = query_masks.unsqueeze(-1).repeat(1, 1, K.size()[1])  # (h*N, T_q, T_k)
        outputs = outputs * query_masks  # broadcasting. (h*N, T_q, T_k)

        # Dropouts
        outputs = self.dropout(outputs)

        attention_weights = outputs.clone()

        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = outputs.split(N, dim=0)
        outputs = torch.cat(outputs, dim=2)  # (N, T_q, C)

        attention_weights = attention_weights.split(N, dim=0)
        attention_weights = torch.cat(attention_weights, dim=2) # (N, T_q, T_k * N)

        return outputs, attention_weights


if __name__ == '__main__':
    features = torch.arange(0, 24)
    features = torch.where(features < 20, features, torch.zeros_like(features))
    features = features.view([2, 3, 4]).float()

    attention1 = Attention1(hidden_dim=features.size()[-1])
    print(attention1(features))

    attention2 = Attention2(hidden_dim=features.size()[-1])
    print(attention2(features))

    attention3 = Multihead_Attention(hidden_dim=features.size()[-1],
                                     num_heads=2,
                                     dropout_rate=0.0)
    print(attention3(features, features, features))
