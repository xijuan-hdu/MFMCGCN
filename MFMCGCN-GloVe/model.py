# coding:utf-8
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from common.tree import head_to_adj
from common.transformer_encoder import TransformerEncoder
from common.RGAT import RGATEncoder
import torch.nn.functional as F
import copy
import math


class MFMCGCNABSA(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        in_dim = args.hidden_dim
        self.args = args
        self.enc = ABSAEncoder(args, emb_matrix=emb_matrix)
        self.classifier = nn.Linear(3 * in_dim, args.num_class)

    def forward(self, inputs):
        hiddens, h_sy, h_se, h_csy, h_cse = self.enc(inputs)
        logits = self.classifier(hiddens)
        return logits, hiddens, h_sy, h_se, h_csy, h_cse


class ABSAEncoder(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        self.args = args
        self.emb_matrix = emb_matrix

        # #################### Embeddings ###################
        self.emb = nn.Embedding(args.tok_size, args.emb_dim, padding_idx=0)
        if emb_matrix is not None:
            self.emb.weight = nn.Parameter(emb_matrix.cuda(), requires_grad=False)
        self.pos_emb = (
            nn.Embedding(args.pos_size, args.pos_dim, padding_idx=0) if args.pos_dim > 0 else None
        )  # POS emb
        self.post_emb = (
            nn.Embedding(args.post_size, args.post_dim, padding_idx=0)
            if args.post_dim > 0
            else None
        )  # position emb

        # #################### Encoder ###################

        if self.args.model.lower() == "rgat":
            self.dep_emb = (
                nn.Embedding(args.dep_size, args.dep_dim, padding_idx=0)
                if args.dep_dim > 0
                else None
            )  # position emb
            embeddings = (self.emb, self.pos_emb, self.post_emb, self.dep_emb)
            self.encoder = DoubleEncoder(
                args, embeddings, args.hidden_dim, args.num_layers, use_dep=True
            )
        else:
            print(
                "Invalid model name {}, it should be (std, GAT, RGAT)".format(
                    self.args.model.lower()
                )
            )
            exit(0)
        self.interGCN = INTERGCN(args)

        self.attn = MultiHeadAttention(args.head_num_GCN, args.rnn_hidden * 2)

        self.h_weight = nn.Parameter(torch.FloatTensor(2).normal_(0.5, 0.5))

        self.gcn1 = GCN(args, args.hidden_dim, args.num_layers_GCN)
        self.gcn2 = GCN(args, args.hidden_dim, args.num_layers_GCN)
        self.gcn_common = GCN(args, args.hidden_dim, args.num_layers_GCN)

        # #################### pooling and fusion modules ###################
        if self.args.pooling.lower() == "attn":
            self.attn = torch.nn.Linear(args.hidden_dim, 1)

        if self.args.output_merge.lower() != "none":
            self.inp_map = torch.nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        if self.args.output_merge.lower() == "none":
            pass
        elif self.args.output_merge.lower() == "attn":
            self.out_attn_map = torch.nn.Linear(args.hidden_dim * 2, 1)
        elif self.args.output_merge.lower() == "gate":
            self.out_gate_map = torch.nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        elif self.args.output_merge.lower() == "gatenorm" or self.args.output_merge.lower() == "gatenorm2":
            self.out_gate_map = torch.nn.Linear(args.hidden_dim * 2, args.hidden_dim)
            self.out_norm = nn.LayerNorm(args.hidden_dim)
        elif self.args.output_merge.lower() == "addnorm":
            self.out_norm = nn.LayerNorm(args.hidden_dim)
        else:
            print("Invalid output_merge type: ", self.args.output_merge)
            exit()
        self.gcn_inp_map = torch.nn.Linear(args.hidden_dim * 3, args.hidden_dim)
        if self.args.output_merge.lower() != "none":
            self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.eye_(self.inp_map.weight)
        torch.nn.init.zeros_(self.inp_map.bias)

    def inputs_to_att_adj(self, input, score_mask):
        attn_tensor = self.attn(input, input, score_mask)  # [batch_size, head_num, seq_len, seq_len]
        attn_tensor = torch.sum(attn_tensor, dim=1)
        attn_tensor = select(attn_tensor, self.args.top_k) * attn_tensor
        return attn_tensor

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask_ori, lengths, aspect_double_index, dependency_graph, aspect_graph, adj1 = inputs  # unpack inputs
        maxlen = max(lengths.data)

        """
        print('tok', tok, tok.size())
        print('asp', asp, asp.size())
        print('pos-tag', pos, pos.size())
        print('head', head, head.size())
        print('deprel', deprel, deprel.size())
        print('postition', post, post.size())
        print('mask', mask, mask.size())
        print('l', l, l.size())
        """
        adj_lst, label_lst = [], []
        for idx in range(len(lengths)):
            adj_i, label_i = head_to_adj(
                maxlen,
                head[idx],
                tok[idx],
                deprel[idx],
                lengths[idx],
                mask_ori[idx],
                directed=self.args.direct,
                self_loop=self.args.loop,
            )
            adj_lst.append(adj_i.reshape(1, maxlen, maxlen))
            label_lst.append(label_i.reshape(1, maxlen, maxlen))

        # [B, maxlen, maxlen]
        adj = np.concatenate(adj_lst, axis=0)
        adj = torch.from_numpy(adj).cuda()

        # [B, maxlen, maxlen]
        labels = np.concatenate(label_lst, axis=0)
        label_all = torch.from_numpy(labels).cuda()


        if self.args.model.lower() == "rgat":
            sent_out, graph_out = self.encoder(
                adj=adj, relation_matrix=label_all, inputs=inputs, lengths=lengths,
            )
            # sent_out, graph_out = self.encoder(
            #     adj=None, relation_matrix=label_all, inputs=inputs, lengths=lengths
            # )
        else:
            print(
                "Invalid model name {}, it should be (std, GAT, RGAT)".format(
                    self.args.model.lower()
                )
            )
            exit(0)
        inter_output = self.interGCN(asp, aspect_double_index, sent_out, dependency_graph, aspect_graph, lengths)

        # ###########pooling and fusion #################
        asp_wn = mask_ori.sum(dim=1).unsqueeze(-1)
        mask = mask_ori.unsqueeze(-1).repeat(1, 1, self.args.hidden_dim)  # mask for h

        score_mask = torch.matmul(sent_out, sent_out.transpose(-2, -1))
        score_mask = (score_mask == 0)
        score_mask = score_mask.unsqueeze(1).repeat(1, self.args.head_num_GCN, 1, 1).cuda()

        att_adj = self.inputs_to_att_adj(sent_out, score_mask)

        h_sy = self.gcn1(adj1, sent_out, score_mask, 'syntax')
        h_se = self.gcn2(att_adj, sent_out, score_mask, 'semantic')
        h_csy = self.gcn_common(adj, sent_out, score_mask, 'syntax')
        h_cse = self.gcn_common(att_adj, sent_out, score_mask, 'semantic')
        h_com = (self.h_weight[0] * h_csy + self.h_weight[1] * h_cse) / 2

        if self.args.pooling.lower() == "avg":  # avg pooling
            graph_out = (graph_out * mask).sum(dim=1) / asp_wn  # masking
            h_sy_mean = (h_sy * mask).sum(dim=1) / asp_wn
            h_se_mean = (h_se * mask).sum(dim=1) / asp_wn
            h_com_mean = (h_com * mask).sum(dim=1) / asp_wn

        elif self.args.pooling.lower() == "max":  # max pooling
            graph_out = torch.max(graph_out * mask, dim=1).values
            h_sy_mean = torch.max(h_sy * mask, dim=1).values
            h_se_mean = torch.max(h_se * mask, dim=1).values
            h_com_mean = torch.max(h_com * mask, dim=1).values

        dym_output = torch.cat((h_sy_mean, h_se_mean, h_com_mean), dim=-1)
        dym_output = F.relu(self.gcn_inp_map(dym_output))
        # elif self.args.pooling.lower() == "attn":
        #     # [B, seq_len, 1]
        #     attns = torch.tanh(self.attn(graph_out))
        #     # print('attn', attns.size())
        #     for i in range(mask_ori.size(0)):
        #         for j in range(mask_ori.size(1)):
        #             if mask_ori[i, j] == 0:
        #                 mask_ori[i, j] = -1e10
        #     masked_attns = F.softmax(mask_ori * attns.squeeze(-1), dim=1)
        #     # print('mask_attns', masked_attns.size())
        #     graph_out = torch.matmul(masked_attns.unsqueeze(1), graph_out).squeeze(1)

        if self.args.output_merge.lower() == "none":
            return graph_out + inter_output + inter_output + dym_output

        sent_out = self.inp_map(sent_out)  # avg pooling
        inter_output = self.inp_map(inter_output)
        if self.args.pooling.lower() == "avg":
            sent_out = (sent_out * mask).sum(dim=1) / asp_wn
            inter_output = (inter_output * mask).sum(dim=1)
        elif self.args.pooling.lower() == "max":  # max pooling
            sent_out = torch.max(sent_out * mask, dim=1).values
            inter_output = torch.max(inter_output * mask, dim=1).values
        graph_out = F.relu(graph_out)
        sent_out = F.relu(sent_out)
        # print(sent_out.size(),graph_out.size())
        # print(inter_output.size())
        if self.args.output_merge.lower() == "gate":  # gate feature fusion
            gate = torch.sigmoid(
                self.out_gate_map(torch.cat([graph_out, sent_out], dim=-1))
            )
            outputs = graph_out * gate + (1 - gate) * sent_out  # + inter_output + dym_output
            # MAMS dataset
            # outputs = torch.cat([inter_output,outputs,dym_output], dim=-1)
            # Laptops dataset
            # outputs = torch.cat([inter_output,0.5 * outputs,dym_output], dim=-1)
            # Restaurants dataset
            # outputs = torch.cat([0.1 * inter_output,0.5 * outputs,dym_output], dim=-1)
            # Twitter dataset
            outputs = torch.cat([inter_output,outputs,dym_output], dim=-1)

        return outputs, h_sy, h_se, h_csy, h_cse


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text.float(), self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class INTERGCN(nn.Module):
    def __init__(self, args):
        super(INTERGCN, self).__init__()
        self.args = args
        self.gc1 = GraphConvolution(2 * args.hidden_dim, 2 * args.hidden_dim)
        self.gc2 = GraphConvolution(2 * args.hidden_dim, 2 * args.hidden_dim)
        self.gc3 = GraphConvolution(2 * args.hidden_dim, 2 * args.hidden_dim)
        self.gc4 = GraphConvolution(2 * args.hidden_dim, 2 * args.hidden_dim)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        # print(seq_len,batch_size)
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        # print(text_len)
        # print(aspect_double_idx)
        for i in range(batch_size):

            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i, 0]):
                weight[i].append(1 - (aspect_double_idx[i, 0] - j) / context_len)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1]):
                weight[i].append(0)
            for j in range(aspect_double_idx[i, 1], text_len[i]):
                weight[i].append(1 - (j - aspect_double_idx[i, 1]) / context_len)

            for j in range(text_len[i], seq_len):
                weight[i].append(0)

        # for i in range(len(weight)):
        #     print(len(weight[i]))
        #     print(weight[i])
        # for i in range(len(weight)):
        #     print(len(weight[i]))
        # print(weight)
        weight = torch.tensor(weight).unsqueeze(2).to(self.args.device)

        return weight * x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i, 0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i, 1] + 1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.args.device)
        return mask * x

    def forward(self, aspect_indices, aspect_double_idx, text_out, adj, d_adj, length):

        # print(length)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        # print(aspect_len)
        # print(aspect_double_idx)
        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, length, aspect_len), adj))
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, length, aspect_len), adj))

        x_d = F.relu(self.gc3(self.position_weight(x, aspect_double_idx, length, aspect_len), d_adj))
        x_d = F.relu(self.gc4(self.position_weight(x_d, aspect_double_idx, length, aspect_len), d_adj))

        x = x + 0.2 * x_d

        # x = self.mask(x, aspect_double_idx)
        # alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        #
        # alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        #
        # x = torch.matmul(alpha, text_out).squeeze(1)

        return x


class DoubleEncoder(nn.Module):
    def __init__(self, args, embeddings, mem_dim, num_layers, use_dep=False):
        super(DoubleEncoder, self).__init__()
        self.args = args
        self.layers = num_layers
        self.mem_dim = mem_dim
        self.in_dim = args.emb_dim + args.post_dim + args.pos_dim
        if use_dep:
            self.emb, self.pos_emb, self.post_emb, self.dep_emb = embeddings
        else:
            self.emb, self.pos_emb, self.post_emb = embeddings

        # Sentence Encoder
        input_size = self.in_dim
        self.Sent_encoder = nn.LSTM(
            input_size,
            args.rnn_hidden,
            args.rnn_layers,
            batch_first=True,
            dropout=args.rnn_dropout,
            bidirectional=args.bidirect,
        )
        if args.bidirect:
            self.in_dim = args.rnn_hidden * 2
        else:
            self.in_dim = args.rnn_hidden

        # dropout
        self.rnn_drop = nn.Dropout(args.rnn_dropout)
        self.in_drop = nn.Dropout(args.input_dropout)

        # Graph Encoder
        if use_dep:
            self.graph_encoder = RGATEncoder(
                num_layers=num_layers,
                d_model=args.rnn_hidden * 2,
                heads=args.attn_heads,
                d_ff=args.rnn_hidden * 2,
                dropout=args.layer_dropout,
                att_drop=args.att_dropout,
                use_structure=True,
                alpha=args.alpha,
                beta=args.beta,
                eps=args.eps,
            )
        else:
            self.graph_encoder = TransformerEncoder(
                num_layers=num_layers,
                d_model=args.rnn_hidden * 2,
                heads=args.attn_heads,
                d_ff=args.rnn_hidden * 2,
                dropout=args.layer_dropout,
            )

        self.out_map = nn.Linear(args.rnn_hidden * 2, args.rnn_hidden)

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(
            batch_size, self.args.rnn_hidden, self.args.rnn_layers, self.args.bidirect
        )
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True, enforce_sorted=False)
        rnn_outputs, (ht, ct) = self.Sent_encoder(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs, lengths, relation_matrix=None):
        tok, asp, pos, head, deprel, post, a_mask, seq_len, _, _, _, _ = inputs  # unpack inputs
        # embedding
        word_embs = self.emb(tok)
        embs = [word_embs]
        if self.args.pos_dim > 0:
            embs += [self.pos_emb(pos)]
        if self.args.post_dim > 0:
            embs += [self.post_emb(post)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # Sentence encoding
        sent_output = self.rnn_drop(
            self.encode_with_rnn(embs, seq_len.cpu(), tok.size()[0])
        )  # [B, seq_len, H]

        mask = adj.eq(0) if adj is not None else None
        key_padding_mask = sequence_mask(lengths) if lengths is not None else None  # [B, seq_len]
        dep_relation_embs = self.dep_emb(relation_matrix) if relation_matrix is not None else None

        # Graph encoding
        inp = sent_output
        graph_output = self.graph_encoder(
            inp, mask=mask, src_key_padding_mask=key_padding_mask, structure=dep_relation_embs,
        )  # [bsz, seq_len, H]
        graph_output = self.out_map(graph_output)
        return sent_output, graph_output


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()


def sequence_mask(lengths, max_len=None):
    """
    create a boolean mask from sequence length `[batch_size, 1, seq_len]`
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) >= (lengths.unsqueeze(1))


class GCN(nn.Module):
    def __init__(self, args, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.args = args
        self.layers = num_layers
        self.mem_dim = mem_dim
        self.in_dim = args.rnn_hidden * 2

        # drop out
        self.in_drop = nn.Dropout(args.input_dropout)
        self.gcn_drop = nn.Dropout(args.gcn_dropout)

        # gcn layer
        self.W = nn.ModuleList()
        self.attn = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.in_dim + layer * self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

            # attention adj layer
            self.attn.append(MultiHeadAttention(args.head_num_GCN, input_dim)) if layer != 0 else None

    def GCN_layer(self, adj, gcn_inputs, denom, l):
        Ax = adj.bmm(gcn_inputs)
        AxW = self.W[l](Ax)
        AxW = AxW / denom
        gAxW = F.relu(AxW) + self.W[l](gcn_inputs)
        # if dataset is not laptops else gcn_inputs = self.gcn_drop(gAxW)
        gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
        return gcn_inputs

    def forward(self, adj, inputs, score_mask, type):
        # gcn
        denom = adj.sum(2).unsqueeze(2) + 1  # norm adj
        out = self.GCN_layer(adj, inputs, denom, 0)
        # 第二层之后gcn输入的adj是根据前一层隐藏层输出求得的

        for i in range(1, self.layers):
            # concat the last layer's out with input_feature as the current input
            inputs = torch.cat((inputs, out), dim=-1)

            if type == 'semantic':
                # att_adj
                adj = self.attn[i - 1](inputs, inputs, score_mask)  # [batch_size, head_num, seq_len, dim]

                if self.args.second_layer == 'max':
                    probability = F.softmax(adj.sum(dim=(-2, -1)), dim=0)
                    max_idx = torch.argmax(probability, dim=1)
                    adj = torch.stack([adj[i][max_idx[i]] for i in range(len(max_idx))], dim=0)
                else:
                    adj = torch.sum(adj, dim=1)

                adj = select(adj, self.args.top_k) * adj
                denom = adj.sum(2).unsqueeze(2) + 1  # norm adj

            out = self.GCN_layer(adj, inputs, denom, i)
        return out


def select(matrix, top_num):
    batch = matrix.size(0)
    len = matrix.size(1)
    matrix = matrix.reshape(batch, -1)
    maxk, _ = torch.topk(matrix, top_num, dim=1)

    for i in range(batch):
        matrix[i] = (matrix[i] >= maxk[i][-1])
    matrix = matrix.reshape(batch, len, len)
    matrix = matrix + matrix.transpose(-2, -1)

    # selfloop
    for i in range(batch):
        matrix[i].fill_diagonal_(1)

    return matrix


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    # d_model:hidden_dim，h:head_num
    def __init__(self, head_num, hidden_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % head_num == 0

        self.d_k = int(hidden_dim // head_num)
        self.head_num = head_num
        self.linears = clones(nn.Linear(hidden_dim, hidden_dim), 2)
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, score_mask, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if score_mask is not None:
            scores = scores.masked_fill(score_mask, -1e9)

        b = ~score_mask[:, :, :, 0:1]
        p_attn = F.softmax(scores, dim=-1) * b.float()
        if dropout is not None:
            p_attn = dropout(p_attn)
        return p_attn

    def forward(self, query, key, score_mask):
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.head_num, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        attn = self.attention(query, key, score_mask, dropout=self.dropout)

        return attn