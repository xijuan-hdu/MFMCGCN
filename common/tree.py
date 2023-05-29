"""
Basic operations on trees.
"""

import numpy as np


def head_to_adj(sent_len, head, tokens, label, len_, mask, directed=False, self_loop=True):
    """
    Convert a sequence of head indexes into a 0/1 matirx and label matrix.
    """
    adj_matrix = np.zeros((sent_len, sent_len), dtype=np.float32)
    label_matrix = np.zeros((sent_len, sent_len), dtype=np.int64)

    assert not isinstance(head, list)
    tokens = tokens[:len_].tolist()
    head = head[:len_].tolist()
    label = label[:len_].tolist()
    # print(head)
    # print('tokens', tokens)
    # print('head', head, len(head))
    # print('label', label)
    # print('mask', mask, len(mask))
    asp_idx = [idx for idx in range(len(mask)) if mask[idx] == 1]
    for idx, head in enumerate(head):
        if idx in asp_idx:
            for k in asp_idx:
                adj_matrix[idx][k] = 1
                label_matrix[idx][k] = 2
        if head != 0:
            adj_matrix[idx, head - 1] = 1
            label_matrix[idx, head - 1] = label[idx]
        else:
            if self_loop:
                adj_matrix[idx, idx] = 1
                label_matrix[idx, idx] = 2
                continue
        if not directed:
            adj_matrix[head - 1, idx] = 1
            label_matrix[head - 1, idx] = label[idx]
        if self_loop:
            adj_matrix[idx, idx] = 1
            label_matrix[idx, idx] = 2

    return adj_matrix, label_matrix

class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self,child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self,'_size'):
            return self._size
        count = 1
        for i in range(self.num_children):#xrange是一个生成器
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self,'_depth'):
            return self._depth
        count = 0
        if self.num_children>0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth>count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x

def head_to_tree(head, tokens, len_):
    """
    Convert a sequence of head indexes into a tree object.
    """
    if isinstance(head, list) == False:
        tokens = tokens[:len_].tolist()
        head = head[:len_].tolist()
    root = None

    nodes = [Tree() for _ in head]

    for i in range(len(nodes)):
        h = head[i]
        nodes[i].idx = i
        nodes[i].dist = -1 # just a filler
        if h == 0:
            root = nodes[i]
        else:
            nodes[h-1].add_child(nodes[i])

    assert root is not None
    return root


def tree_to_adj(sent_len, tree, directed=False, self_loop=True):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)

    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]

        idx += [t.idx]

        for c in t.children:
            ret[t.idx, c.idx] = 1
        queue += t.children

    if not directed:
        ret = ret + ret.T

    if self_loop:
        for i in idx:
            ret[i, i] = 1

    return ret