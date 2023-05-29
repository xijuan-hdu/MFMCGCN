# coding:utf-8
import sys

sys.path.append('../')
import torch
import numpy as np
import torch.nn as nn
from common.tree import head_to_adj
from common.transformer_encoder import TransformerEncoder
from common.RGAT import RGATEncoder
from transformers import BertModel, BertConfig, BertTokenizer
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
import copy
import math
from torch.autograd import Variable

bert_config = BertConfig.from_pretrained("bert-base-uncased")
# bert_config.output_hidden_states = True
bert_config.num_labels = 3
bert = BertModel.from_pretrained("bert-base-uncased", config=bert_config)


class MFMCGCNABSA(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        # in_dim = args.emb_dim + args.bert_out_dim
        # in_dim = 1 * args.hidden_dim + args.emb_dim
        in_dim = 3 * args.hidden_dim
        self.args = args
        self.enc = ABSAEncoder(args)
        self.classifier = nn.Linear(in_dim, args.num_class)
        self.dropout = nn.Dropout(0.3)

    def forward(self, inputs):
        outputs = self.enc(inputs)
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)
        return logits, outputs


class ABSAEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.pos_emb = (
            nn.Embedding(args.pos_size, args.pos_dim, padding_idx=0) if args.pos_dim > 0 else None
        )  # pos tag emb

        self.post_emb = (
            nn.Embedding(args.post_size, args.post_dim, padding_idx=0)
            if args.post_dim > 0
            else None
        )  # position emb

        if self.args.model.lower() in ["std", "gat"]:
            embs = (self.pos_emb, self.post_emb)
            self.encoder = DoubleEncoder(args, embeddings=embs, use_dep=True)
        elif self.args.model.lower() == "rgat":
            self.dep_emb = (
                nn.Embedding(args.dep_size, args.dep_dim, padding_idx=0)
                if args.dep_dim > 0
                else None
            )  # position emb
            embs = (self.pos_emb, self.post_emb, self.dep_emb)
            self.encoder = DoubleEncoder(args, embeddings=embs, use_dep=True)

        self.interGCN = INTERGCN(args)

        self.attn = MultiHeadAttention(args.head_num_GCN, args.rnn_hidden * 2)

        self.h_weight = nn.Parameter(torch.FloatTensor(2).normal_(0.5, 0.5))

        self.gcn1 = GCN(args, args.hidden_dim, args.num_layers_GCN)
        self.gcn2 = GCN(args, args.hidden_dim, args.num_layers_GCN)
        self.gcn_common = GCN(args, args.hidden_dim, args.num_layers_GCN)

        self.gcn_inp_map = torch.nn.Linear(args.hidden_dim * 3, args.hidden_dim)

        self.inp_map = torch.nn.Linear(args.hidden_dim * 3, args.hidden_dim)
        self.inp_map1 = torch.nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        if self.args.output_merge.lower() == "gate":
            self.gate_map = nn.Linear(args.rnn_hidden * 2, args.rnn_hidden)
        elif self.args.output_merge.lower() == "none":
            pass
        else:
            print('Invalid output_merge type !!!')
            exit()

    def inputs_to_att_adj(self, input, score_mask):
        attn_tensor = self.attn(input, input, score_mask)  # [batch_size, head_num, seq_len, seq_len]
        attn_tensor = torch.sum(attn_tensor, dim=1)
        attn_tensor = select(attn_tensor, self.args.top_k) * attn_tensor
        return attn_tensor

    def forward(self, inputs):
        (
            tok,
            asp,
            pos,
            head,
            deprel,
            post,
            mask,
            l,
            word_idx,
            segment_ids,
            aspect_double_index,
            dependency_graph,
            aspect_graph,
            adj1,
            left_indicies1,
            aspect_indices,
            left_indicies,
            text_indices,
            asp1,
            label,
        ) = inputs  # unpack inputs
        maxlen = max(l.data)


        adj_lst, label_lst = [], []
        for idx in range(len(l)):
            adj_i, label_i = head_to_adj(
                maxlen,
                head[idx],
                tok[idx],
                deprel[idx],
                l[idx],
                mask[idx],
                directed=self.args.direct,
                self_loop=self.args.loop,
            )
            adj_lst.append(adj_i.reshape(1, maxlen, maxlen))
            label_lst.append(label_i.reshape(1, maxlen, maxlen))

        adj = np.concatenate(adj_lst, axis=0)  # [B, maxlen, maxlen]
        adj = torch.from_numpy(adj).cuda()

        labels = np.concatenate(label_lst, axis=0)  # [B, maxlen, maxlen]

        label_all = torch.from_numpy(labels).cuda()

        if self.args.model.lower() == "rgat":
            h = self.encoder(
                adj=adj, relation_matrix=label_all, inputs=inputs, lengths=l
            )
        else:
            print(
                "Invalid model name {}, it should be (std, GAT, RGAT)".format(
                    self.args.model.lower()
                )
            )
            exit(0)

        graph_out, bert_pool_output, sent_out, bert_out = h[0], h[1], h[2], h[3]
        asp_wn = mask.sum(dim=1).unsqueeze(-1)  # aspect words num
        mask = mask.unsqueeze(-1).repeat(1, 1, self.args.rnn_hidden)  # mask for h

        graph_enc_outputs = (graph_out * mask).sum(dim=1) / asp_wn  # mask h

        score_mask = torch.matmul(sent_out, sent_out.transpose(-2, -1))
        score_mask = (score_mask == 0)
        score_mask = score_mask.unsqueeze(1).repeat(1, self.args.head_num_GCN, 1, 1).cuda()

        att_adj = self.inputs_to_att_adj(sent_out, score_mask)


        h_sy = self.gcn1(adj1, sent_out, score_mask, 'syntax')
        h_se = self.gcn2(att_adj, sent_out, score_mask, 'semantic')
        h_csy = self.gcn_common(adj, sent_out, score_mask, 'syntax')
        h_cse = self.gcn_common(att_adj, sent_out, score_mask, 'semantic')
        h_com = (self.h_weight[0] * h_csy + self.h_weight[1] * h_cse) / 2
        h_sy_mean = (h_sy * mask).sum(dim=1) / asp_wn
        h_se_mean = (h_se * mask).sum(dim=1) / asp_wn
        h_com_mean = (h_com * mask).sum(dim=1) / asp_wn
        dym_output = torch.cat((h_sy_mean, h_se_mean, h_com_mean), dim=-1)
        dym_output = F.relu(self.gcn_inp_map(dym_output))

        inter_output = self.interGCN(asp1, aspect_double_index, sent_out, dependency_graph, aspect_graph, l)
        sent_out = self.inp_map1(sent_out)
        inter_output = self.inp_map1(inter_output)

        bert_enc_outputs = (sent_out * mask).sum(dim=1) / asp_wn
        inter_output = (inter_output * mask).sum(dim=1) / asp_wn
        graph_enc_outputs = F.relu(graph_enc_outputs)
        bert_enc_outputs = F.relu(bert_enc_outputs)
        if self.args.output_merge.lower() == "none":
            merged_outputs = graph_enc_outputs
        elif self.args.output_merge.lower() == "gate":
            # inter_output = self.dropout1(inter_output)
            # dym_output = self.dropout2(dym_output)
            gate = torch.sigmoid(self.gate_map(torch.cat([graph_enc_outputs, bert_enc_outputs], 1)))
            merged_outputs = gate * graph_enc_outputs + (1 - gate) * bert_enc_outputs
            # merged_outputs = self.dropout3(merged_outputs)
            # MAMS dataset
            # outputs = torch.cat([inter_output,outputs,dym_output], dim=-1)
            # Laptops dataset
            # outputs = torch.cat([0.6 * inter_output,outputs,dym_output], dim=-1)
            # Restaurants dataset
            # outputs = torch.cat([0.6 * inter_output,outputs,dym_output], dim=-1)
            # Twitter dataset
            outputs = torch.cat([0.6 * inter_output,merged_outputs,dym_output], dim=-1)
            # outputs = self.inp_map(outputs)
            # outputs = inter_output + merged_outputs + dym_output
        else:
            print('Invalid output_merge type !!!')
            exit()

        # outputs = torch.cat([outputs,bert_pool_output], 1)
        return outputs


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
        self.gc1 = GraphConvolution(args.hidden_dim * 2, args.hidden_dim * 2)
        self.gc2 = GraphConvolution(args.hidden_dim * 2, args.hidden_dim * 2)
        self.gc3 = GraphConvolution(args.hidden_dim * 2, args.hidden_dim * 2)
        self.gc4 = GraphConvolution(args.hidden_dim * 2, args.hidden_dim * 2)
        # self.gc1 = inter_GCN(args, args.hidden_dim, args.num_layers_GCN)
        # self.gc2 = inter_GCN(args, args.hidden_dim, args.num_layers_GCN)
        # self.gc3 = inter_GCN(args, args.hidden_dim, args.num_layers_GCN)
        # self.gc4 = inter_GCN(args, args.hidden_dim, args.num_layers_GCN)
        # self.inmap = nn.Linear(args.hidden_dim,args.hidden_dim * 2)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        aspect_double_idx = aspect_double_idx.cpu().numpy()

        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]

        for i in range(batch_size):

            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i, 0]):
                weight[i].append(1 - (aspect_double_idx[i, 0] - j) / context_len)
            for j in range(aspect_double_idx[i, 0], min(aspect_double_idx[i, 1], self.args.max_len)):
                weight[i].append(0)
            for j in range(min(aspect_double_idx[i, 1], self.args.max_len), text_len[i]):
                weight[i].append(1 - (j - aspect_double_idx[i, 1]) / context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)

        weight = torch.tensor(weight).unsqueeze(2).to(self.args.device)

        return weight * x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i, 0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i, 0], min(aspect_double_idx[i, 1], self.args.max_len)):
                mask[i].append(1)
            for j in range(min(aspect_double_idx[i, 1], self.args.max_len), seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.args.device)
        return mask * x

    def forward(self, aspect_indices, aspect_double_idx, text_out, adj, d_adj, length):


        aspect_len = torch.sum(aspect_indices != 0, dim=-1)

        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, length, aspect_len), adj))
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, length, aspect_len), adj))

        x_d = F.relu(self.gc3(self.position_weight(x, aspect_double_idx, length, aspect_len), d_adj))
        x_d = F.relu(self.gc4(self.position_weight(x_d, aspect_double_idx, length, aspect_len), d_adj))

        x = x + 0.2 * x_d


        return x


class DoubleEncoder(nn.Module):
    def __init__(self, args, embeddings=None, use_dep=False):
        super(DoubleEncoder, self).__init__()
        self.args = args
        self.bert = bert


        for param in self.bert.parameters():
            param.requires_grad = True

        self.dropout_bert = nn.Dropout(bert_config.hidden_dropout_prob)

        self.in_drop = nn.Dropout(args.input_dropout)
        self.dense = nn.Linear(args.hidden_dim, args.bert_out_dim)  # dimension reduction
        self.in_dim = args.emb_dim + args.post_dim + args.pos_dim
        self.rnn = nn.LSTM(self.in_dim, args.rnn_hidden, args.rnn_layers, batch_first=True, bidirectional=True)
        if args.bidirect:
            self.in_dim = args.rnn_hidden * 2
        else:
            self.in_dim = args.rnn_hidden
        self.rnn_drop = nn.Dropout(args.rnn_dropout)

        if use_dep:
            self.pos_emb, self.post_emb, self.dep_emb = embeddings
            self.Graph_encoder = RGATEncoder(
                num_layers=args.num_layer,
                d_model=args.rnn_hidden * 2,
                heads=args.attn_heads,
                d_ff=args.hidden_dim,
                dep_dim=self.args.dep_dim,
                att_drop=self.args.att_dropout,
                dropout=args.layer_dropout,
                use_structure=True
            )
        else:
            self.pos_emb, self.post_emb = embeddings
            self.Graph_encoder = TransformerEncoder(
                num_layers=args.num_layer,
                d_model=args.bert_out_dim,
                heads=4,
                d_ff=args.hidden_dim,
                dropout=0.0
            )
        if args.reset_pooling:
            self.reset_params(bert.pooler.dense)
        self.out_map = nn.Linear(args.rnn_hidden * 2, args.rnn_hidden)

    def reset_params(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    def create_bert_embs(self, tok, pos, post, word_idx, segment_ids):
        outputs = self.bert(tok, token_type_ids=segment_ids)
        feature_output = outputs[0]
        bert_pool_output = outputs[1]
        word_embs = torch.stack([torch.index_select(f, 0, w_i)
                                 for f, w_i in zip(feature_output, word_idx)])
        embs = [word_embs]
        if self.args.pos_dim > 0:
            embs += [self.pos_emb(pos)]
        if self.args.post_dim > 0:
            embs += [self.post_emb(post)]
        embs = torch.cat(embs, dim=-1)
        embs = self.in_drop(embs)
        return embs, bert_pool_output, feature_output

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(
            batch_size, self.args.rnn_hidden, self.args.rnn_layers, self.args.bidirect
        )
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens.cpu(), batch_first=True,
                                                       enforce_sorted=False)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs, lengths, relation_matrix=None):
        (
            tok,
            asp,
            pos,
            head,
            deprel,
            post,
            mask,
            l,
            word_idx,
            segment_ids,
            aspect_double_index,
            dependency_graph,
            aspect_graph,
            adj1,
            _,
            _,
            _,
            _,
            _,
            label,

        ) = inputs  # unpack inputs

        embs, bert_pool_out, bert_out = self.create_bert_embs(tok, pos, post, word_idx, segment_ids)
        rnn_hidden = self.rnn_drop(self.encode_with_rnn(embs, l, embs.size(0)))  # [batch_size, seq_len, hidden]

        # input()

        # bert_out = self.in_drop(bert_out)
        # bert_out = bert_out[:, 0:max(l), :]
        # bert_out = self.dense(bert_out)

        if adj is not None:
            mask = adj.eq(0)
        else:
            mask = None

        if lengths is not None:
            key_padding_mask = sequence_mask(l)  # [B, seq_len]

        if relation_matrix is not None:
            dep_relation_embs = self.dep_emb(relation_matrix)
        else:
            dep_relation_embs = None

        inp = rnn_hidden  # [bsz, seq_len, H]
        graph_out = self.Graph_encoder(
            inp, mask=mask, src_key_padding_mask=key_padding_mask, structure=dep_relation_embs
        )
        graph_out = self.out_map(graph_out)
        return graph_out, bert_pool_out, rnn_hidden, bert_out


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


class inter_GCN(nn.Module):
    def __init__(self, args, mem_dim, num_layers):
        super(inter_GCN, self).__init__()
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


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()


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
