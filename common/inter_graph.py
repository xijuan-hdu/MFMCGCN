import numpy as np
import spacy
import pickle
import json

def dependency_adj_matrix(text, aspect, position, other_aspects,seq_len):
    text_list = text.split()
    # seq_len = len(text_list)
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    position = int(position)
    flag = 1

    for i in range(seq_len):
        word = text_list[i]
        if word in aspect:
            for other_a in other_aspects:
                other_p = int(other_aspects[other_a])
                add = 0
                for other_w in other_a.split():
                    weight = 1 + (1 / (abs(add + other_p - position) + 1))
                    matrix[i][other_p + add] = weight
                    matrix[other_p + add][i] = weight
                    add += 1
    return matrix

def process(filename):
    with open(filename, "r") as infile:
        data = json.load(infile)

    idx2graph = {}
    fout = open(filename+'.graph_inter', 'wb')
    graph_idx = 0
    for d in data:
        tok = list(d["token"])
        text = ""
        for t in tok:
            text += t.lower() + ' '
        text = text.lower().strip()
        length =  len(tok)
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
        for aspect, position in zip(asp_list, position_list):
            aspect_positions[aspect] = position
        for aspect, position in zip(asp_list, position_list):
            aspect = aspect.lower().strip()
            other_aspects = aspect_positions.copy()
            del other_aspects[aspect]
            adj_matrix = dependency_adj_matrix(text, aspect, position, other_aspects,length)
            idx2graph[graph_idx] = adj_matrix
            graph_idx += 1
    pickle.dump(idx2graph, fout)
    print('done !!!'+filename)
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