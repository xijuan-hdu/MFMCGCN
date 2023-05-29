# coding:utf-8
import sys
sys.path.append("..") 
import json
import torch
import numpy as np
from transformers import BertTokenizer
from common.tree import *
import pickle

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def pad_and_truncate(sequence, maxlen, dtype="int64", padding="post", truncating="post", value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == "pre":
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == "post":
        x[: len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x

class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)

class ABSADataLoader(object):
    def __init__(self, filename, batch_size, args, vocab, shuffle=True):
        self.batch_size = batch_size
        self.args = args
        self.vocab = vocab
        self.tokenizer4Inter = Tokenizer4Bert(args.max_len,'bert-base-uncased')
        with open(filename) as infile:
            data = json.load(infile)
        self.raw_data = data

        # preprocess data
        data = self.preprocess(data, vocab, args,filename)
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
                text = " ".join(tok)
                # aspect
                asp = list(aspect["term"])
                aspect_cur = ""
                for i in asp:
                    aspect_cur += i + ' '
                aspect_cur.lower().strip()
                asp_len = len(asp)
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
                text_left, _, text_right = [s.lower().strip() for s in text.partition(aspect_cur)]

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
                aspect_indices = self.tokenizer4Inter.text_to_sequence(aspect_cur)
                text_indices = self.tokenizer4Inter.text_to_sequence(text)
                left_indices = self.tokenizer4Inter.text_to_sequence(text_left)
                tok, asp1, segment_ids, word_idx, aspect_idx = self.convert_features_bert(tok, asp)
                left_bert_indices = self.tokenizer4Inter.text_to_sequence("[CLS] " + text_left + " [SEP]")
                label = pol_vocab.stoi.get(label)

                # mapping pos
                pos = [pos_vocab.stoi.get(t, pos_vocab.unk_index) for t in pos]
                asp = [token_vocab.stoi.get(t, token_vocab.unk_index) for t in asp]
                # mapping head to int
                head = [int(x) for x in head]
                assert any([x == 0 for x in head])
                # mapping deprel
                deprel = [dep_vocab.stoi.get(t, dep_vocab.unk_index) for t in deprel]
                # mapping post
                post = [post_vocab.stoi.get(t, post_vocab.unk_index) for t in post]

                dependency_graph = idx2graph[graph_id]
                aspect_graph = idx2graph_a[graph_id]
                assert any([x == 0 for x in head])



                aspect_double_index = [aspect["from"], aspect["to"]]
                assert len(pos) == length \
                       and len(head) == length \
                       and len(post) == length \
                       and len(mask) == length
                graph_id += 1
                processed += [
                    (
                        tok, asp1, pos, head, deprel, post, mask, length, word_idx, segment_ids,
                        aspect_double_index,
                        dependency_graph,
                        aspect_graph,
                        left_bert_indices,
                        aspect_indices,
                        left_indices,
                        text_indices,
                        asp,
                        label,

                    )
                ]
        return processed

    def convert_features_bert(self, sentence, aspect):
        """
        BERT features.
        convert sentence to feature.
        """
        cls_token = "[CLS]"
        sep_token = "[SEP]"
        pad_token = 0

        tokens = []
        word_indexer = []
        aspect_tokens = []
        aspect_indexer = []

        for word in sentence:
            word_tokens = self.args.tokenizer.tokenize(word)
            token_idx = len(tokens)
            tokens.extend(word_tokens)
            # word_indexer is for indexing after bert, feature back to the length of original length.
            word_indexer.append(token_idx)

        # aspect
        for word in aspect:
            word_aspect_tokens = self.args.tokenizer.tokenize(word)
            token_idx = len(aspect_tokens)
            aspect_tokens.extend(word_aspect_tokens)
            aspect_indexer.append(token_idx)


        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0

        tokens = [cls_token] + tokens + [sep_token]
        aspect_tokens = [cls_token] + aspect_tokens + [sep_token]
        word_indexer = [i + 1 for i in word_indexer]
        aspect_indexer = [i + 1 for i in aspect_indexer]


        input_ids = self.args.tokenizer.convert_tokens_to_ids(tokens)
        input_aspect_ids = self.args.tokenizer.convert_tokens_to_ids(
            aspect_tokens)

        # check len of word_indexer equals to len of sentence.
        assert len(word_indexer) == len(sentence)
        assert len(aspect_indexer) == len(aspect)

        # 句子后面拼上aspect, segment_idx是bert中用于表示句子idx的标志
        input_cat_ids = input_ids + input_aspect_ids[1:]
        segment_ids = [0] * len(input_ids) + [1] * len(input_aspect_ids[1:])

        # tok, asp, seg, word_idx, asp_idx
        return input_cat_ids, input_aspect_ids, segment_ids, word_indexer, aspect_indexer

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
        asp1 = get_long_tensor(batch[1], batch_size)
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
        # print(length)
        maxlen = max(length)
        word_idx = get_long_tensor(batch[8], batch_size).cuda()
        segment_ids = get_long_tensor(batch[9], batch_size).cuda()

        aspect_double_index = get_long_tensor(batch[10], batch_size)
        #  dep graph
        dependency_graph = get_float_tensor_graph(batch[11], batch_size,maxlen)
        #  asp graph
        aspect_graph = get_float_tensor_graph(batch[12], batch_size,maxlen)
        left_bert_indices = get_long_tensor(batch[13],batch_size)
        aspect_indices = get_long_tensor(batch[14],batch_size)
        left_indices = get_long_tensor(batch[15],batch_size)
        text_indices = get_long_tensor(batch[16],batch_size)
        asp = get_long_tensor(batch[17],batch_size)
        def inputs_to_tree_reps(maxlen, head, words, l):
            trees = [head_to_tree(head[i], words[i], l[i]) for i in range(l.size(0))]
            adj = [tree_to_adj(maxlen, tree, directed=self.args.direct, self_loop=True).reshape(1, maxlen,maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            return adj


        adj = torch.tensor(inputs_to_tree_reps(maxlen, head, tok, length)).cuda()
       

        # label
        label = torch.LongTensor(batch[-1])
        # bert input

        return (
            tok,
            asp,
            pos,
            head,
            deprel,
            post,
            mask,
            length,
            word_idx,
            segment_ids,
            aspect_double_index,
            dependency_graph,
            aspect_graph,
            adj,
            left_bert_indices,
            aspect_indices,
            left_indices,
            text_indices,
            asp,
            label,
        )

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

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else 1 for t in tokens] # the id of [UNK] is ``1''
    return ids
def get_float_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded FloatTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.FloatTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, : len(s)] = torch.FloatTensor(s)
    return tokens

def get_float_tensor_graph(tokens_list, batch_size,maxlen):
    """ Convert list of list of tokens to a padded LongTensor. """
    # token_len = max(len(x) for x in tokens_list)

    tokens = torch.FloatTensor(batch_size, maxlen, maxlen).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, : len(s), : len(s)] = torch.FloatTensor(s)

    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

