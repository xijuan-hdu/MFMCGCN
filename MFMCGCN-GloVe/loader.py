# coding:utf-8
import sys
sys.path.append("..") 
import json
import torch
import numpy as np
import os
import pickle
from common.tree import *

class ABSADataLoader(object):
    def __init__(self, filename, batch_size, args, vocab, shuffle=True):
        self.batch_size = batch_size
        self.args = args
        self.vocab = vocab

        assert os.path.exists(filename), filename
        with open(filename, "r") as infile:
            data = json.load(infile)
        self.raw_data = data

        # preprocess data
        data = self.preprocess(data, vocab, args,filename)

        print("{} instances loaded from {}".format(len(data), filename))

        if shuffle:
            indices = np.arange(len(data))
            np.random.shuffle(indices)
            data = [data[idx] for idx in indices]
        # labels
        pol_vocab = vocab[-1]
        self.labels = [pol_vocab.itos[d[-1]] for d in data]

        # example num
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, vocab, args,filename):
        # unpack vocab
        token_vocab, post_vocab, pos_vocab, dep_vocab, pol_vocab = vocab
        processed = []
        fin = open(filename + '.graph_af', 'rb')
        idx2graph = pickle.load(fin)
        fin.close()
        fin = open(filename + '.graph_inter', 'rb')
        idx2graph_a = pickle.load(fin)
        fin.close()
        graph_id = 0
        for d in data:
            for aspect in d["aspects"]:
                # word token
                tok = list(d["token"])
                if args.lower:
                    tok = [t.lower() for t in tok]

                # aspect
                asp = list(aspect["term"])
                # label
                label = aspect["polarity"]
                # pos_tag
                pos = list(d["pos"])
                # head
                head = list(d["head"])
                # deprel
                deprel = list(d["deprel"])
                # real length
                length = len(tok)
                if(length <= 0):
                    continue;
                # position
                post = (
                    [i - aspect["from"] for i in range(aspect["from"])]
                    + [0 for _ in range(aspect["from"], aspect["to"])]
                    + [i - aspect["to"] + 1 for i in range(aspect["to"], length)]
                )
                # aspect mask
                if len(asp) == 0:
                    mask = [1 for _ in range(length)]  # for rest16
                else:
                    mask = (
                        [0 for _ in range(aspect["from"])]
                        + [1 for _ in range(aspect["from"], aspect["to"])]
                        + [0 for _ in range(aspect["to"], length)]
                    )

                # mapping token
                # print('tok', tok)
                # print('asp', asp)
                tok = [token_vocab.stoi.get(t, token_vocab.unk_index) for t in tok]
                # mapping aspect
                asp = [token_vocab.stoi.get(t, token_vocab.unk_index) for t in asp]
                # mapping label
                label = pol_vocab.stoi.get(label)
                # mapping pos
                pos = [pos_vocab.stoi.get(t, pos_vocab.unk_index) for t in pos]
                # mapping head to int

                head = [int(x) for x in head]
                dependency_graph = idx2graph[graph_id]
                aspect_graph = idx2graph_a[graph_id]
                assert any([x == 0 for x in head])
                # mapping deprel
                deprel = [dep_vocab.stoi.get(t, dep_vocab.unk_index) for t in deprel]
                # mapping post
                post = [post_vocab.stoi.get(t, post_vocab.unk_index) for t in post]
                aspect_double_index = [aspect["from"], aspect["to"]]
                assert (
                    len(tok) == length
                    and len(pos) == length
                    and len(head) == length
                    and len(deprel) == length
                    and len(post) == length
                    and len(mask) == length
                )
                graph_id += 1
                processed += [(tok, asp, pos, head, deprel, post, mask, length,aspect_double_index,dependency_graph,aspect_graph,label)]

        return processed

    def gold(self):
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError

        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # convert to tensors
        # token
        tok = get_long_tensor(batch[0], batch_size)
        # aspect
        asp = get_long_tensor(batch[1], batch_size)
        # pos
        pos = get_long_tensor(batch[2], batch_size)
        # head
        head = get_long_tensor(batch[3], batch_size)
        # deprel
        deprel = get_long_tensor(batch[4], batch_size)
        # post
        post = get_long_tensor(batch[5], batch_size)
        # mask
        mask = get_float_tensor(batch[6], batch_size)
        # length
        length = torch.LongTensor(batch[7])
        # label
        label = torch.LongTensor(batch[-1])
        # left indices
        aspect_double_index = get_long_tensor(batch[8], batch_size)
        #  dep graph
        dependency_graph = get_float_tensor_graph(batch[9], batch_size)
        #  asp graph
        aspect_graph = get_float_tensor_graph(batch[10], batch_size)

        def inputs_to_tree_reps(maxlen, head, words, l):
            trees = [head_to_tree(head[i], words[i], l[i]) for i in range(l.size(0))]
            adj = [tree_to_adj(maxlen, tree, directed=self.args.direct, self_loop=True).reshape(1, maxlen,maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            return adj

        maxlen = max(length)
        adj = torch.tensor(inputs_to_tree_reps(maxlen, head, tok, length)).cuda()
        return (tok, asp, pos, head, deprel, post, mask, length, aspect_double_index,dependency_graph,aspect_graph,adj,label)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, : len(s)] = torch.LongTensor(s)
    return tokens


def get_float_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded FloatTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.FloatTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, : len(s)] = torch.FloatTensor(s)
    return tokens


def get_float_tensor_graph(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)

    tokens = torch.FloatTensor(batch_size, token_len, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, : len(s), : len(s)] = torch.FloatTensor(s)

    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]
