import torch
import torch.nn as nn

import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=True)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.001)
            # self.bias = self.bias * 1e42
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.spmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        out = F.dropout(out,0.9)
        return self.act(out)



class Mp_encoder(nn.Module):
    def __init__(self, P, hidden_dim, attn_drop, mp_ngcn):
        super(Mp_encoder, self).__init__()
        self.P = P
        self.node_level = nn.ModuleList([GCN(hidden_dim, hidden_dim) for _ in range(mp_ngcn)])
        self.gat_heads = 4
        self.dropout = 0.
        """
        每个节点类型只有一个meta path，所以不需要meta path级的attention
        """
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.feat_drop = nn.Dropout(0.2)
        self.mp_ngcn = mp_ngcn

    def forward(self, h, mps, mp_edge):
        embeds = []

        embeds.append(h)
        for i in range(self.mp_ngcn):
            embeds.append(self.node_level[i](embeds[-1], mps[1]))
        z_mp = embeds[-1]


        return z_mp
