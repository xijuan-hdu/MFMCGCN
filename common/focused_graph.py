import pickle

import numpy as np
import spacy
import json

nlp = spacy.load('en_core_web_sm')
def dependency_adj_matrix(text,aspect,position,seq_len,directed=False, self_loop=True):


    document = nlp(text)
    # seq_len = len(text.split())
    matrix = np.zeros((seq_len,seq_len)).astype('float32')
    text_list = text.split()

    for token in document:
        if str(token) in aspect:
            weight = 1
            if(token.i < seq_len):
                for j in range(seq_len):
                    if text_list[j] in aspect:
                        sub_weight = 1
                    else:
                        sub_weight = 1 /(abs(j - int(position)) + 1)
                    matrix[token.i][j] = 1 * sub_weight
                    if not directed:
                        matrix[j][token.i] = 1 * sub_weight
            else:
                weight = 1 / (abs(token.i - int(position)) + 1)
            if token.i < seq_len:
                if self_loop:
                    matrix[token.i][token.i] = 1
                for child in token.children:
                    if str(child) in aspect:
                        weight += 1
                    else:
                        weight += 1 / (abs(child.i - int(position)) + 1)
                    if child.i < seq_len:
                        matrix[token.i][child.i] += 1 * weight
                        if self_loop:
                            matrix[child.i][token.i] += 1 * weight

    return  matrix

def get_con_adj_matrix(aspect, position, aspect_graphs, other_aspects):
    adj_matrix = aspect_graphs[aspect]  ## aspect-focused syntactical dependency adjacency matrix
    position = int(position)
    if len(aspect_graphs) == 1:
        return adj_matrix
    for other_a in other_aspects:
        other_p = int(other_aspects[other_a])
        other_m = aspect_graphs[other_a]
        alpha = 1 / (abs(position - other_p) + 1)
        weight = 1 / len(aspect_graphs)
        adj_matrix += alpha * weight * other_m
    return adj_matrix


def process(filename):

    with open(filename, "r") as infile:
        data = json.load(infile)
    idx2graph = {}
    fout = open(filename + '.graph_af','wb')
    graph_idx = 0
    for d in data:
        # word token
        tok = list(d["token"])
        text = ""
        for t in tok:
            text += t.lower() + ' '
        text = text.lower().strip()
        length = len(tok)
        asp_list = []
        position_list = []

        for aspect in d["aspects"]:
            # aspect
            aspect_cur = ""
            for term in aspect['term']:
                aspect_cur += term + " "
            aspect_cur = aspect_cur.lower().strip()
            asp_list.append(aspect_cur)
            # position
            position_list.append(aspect['from'])
        aspect_graphs = {}
        aspect_positions = {}
        for aspect,position in zip(asp_list,position_list):
            aspect_positions[aspect] = position
        for aspect,position in zip(asp_list,position_list):
            other_aspects = aspect_positions.copy()
            aspect = aspect.lower().strip()
            del other_aspects[aspect]
            adj_matrix = dependency_adj_matrix(text,aspect,position,length)
            aspect_graphs[aspect] = adj_matrix
        for aspect,position in zip(asp_list,position_list):
            aspect = aspect.lower().strip()
            other_aspects = aspect_positions.copy()
            del other_aspects[aspect]
            adj_matrix = get_con_adj_matrix(aspect,position,aspect_graphs,other_aspects)
            idx2graph[graph_idx] = adj_matrix
            graph_idx += 1
    pickle.dump(idx2graph,fout)
    print('done!' + filename)
    fout.close()

if __name__ == '__main__':
    # process('/home/g21tka10/test/dataset/Biaffine/glove/Laptops/test.json')
    # process('/home/g21tka10/test/dataset/Biaffine/glove/Laptops/train.json')
    # process('/home/g21tka10/test/dataset/Biaffine/glove/Laptops/valid.json')
    # process('/home/g21tka10/test/dataset/Biaffine/glove/MAMS/test.json')
    # process('/home/g21tka10/test/dataset/Biaffine/glove/MAMS/train.json')
    # process('/home/g21tka10/test/dataset/Biaffine/glove/MAMS/valid.json')
    # process('/home/g21tka10/test/dataset/Biaffine/glove/Restaurants/train.json')
    # process('/home/g21tka10/test/dataset/Biaffine/glove/Restaurants/test.json')
    # process('/home/g21tka10/test/dataset/Biaffine/glove/Restaurants/valid.json')
    # process('/home/g21tka10/test/dataset/Biaffine/glove/Tweets/train.json')
    # process('/home/g21tka10/test/dataset/Biaffine/glove/Tweets/test.json')
    # process('/home/g21tka10/test/dataset/Biaffine/glove/Tweets/valid.json')
    # process('/home/g21tka10/test/dataset/Biaffine/glove/Laptops/case_study1.json')
    process('/home/g21tka10/test/dataset/Biaffine/glove/Restaurants/multi_aspect.json')