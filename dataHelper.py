# -*- coding:utf-8 -*-
import os
from nltk.tokenize import word_tokenize
import keras as kr
from sentiment import *

POS_PATH = 'D:\\DataSet\\IMDB\\aclImdb\\train\\pos\\pos_all.txt'
NEG_PATH = 'D:\\DataSet\\IMDB\\aclImdb\\train\\neg\\neg_all.txt'
TEST_POS_PATH = 'D:\\DataSet\\IMDB\\aclImdb\\test\\pos\\pos_all.txt'
TEST_NEG_PATH = 'D:\\DataSet\\IMDB\\aclImdb\\test\\neg\\neg_all.txt'
VOCAB_PATH = 'D:\\DataSet\\IMDB\\aclImdb\\train\\vocab.txt'

"""
def read_file(pos_path, neg_path, num = 5000):
    texts = []
    labels = []

    count = 0
    # read positive text
    for file in os.listdir(pos_path):
        count += 1
        if count > num:
            break
        f = open(pos_path + '\\' + file, 'r', encoding='utf-8')
        textlist = []
        for line in f:
            textlist.append(line)
        text = ' '.join(textlist)
        texts.append(text)
        labels.append('pos')
        f.close()

    # read negative text
    count = 0
    for file in os.listdir(neg_path):
        count += 1
        if count > num:
            break
        f = open(neg_path + '\\' + file, 'r', encoding='utf-8')
        for line in f:
            text = " ".join(line)
        texts.append(text)
        labels.append('neg')
        f.close()
    return texts, labels
"""


def read_file(pos_path, neg_path, num):
    texts = []
    labels = []

    f1 = open(pos_path, 'r', encoding='utf-8')
    f2 = open(neg_path, 'r', encoding='utf-8')

    # load positive data
    count = 0
    for line in f1:
        count += 1
        if count > num:
            break
        texts.append(line)
        labels.append('pos')
    f1.close()

    # load negtive data
    count = 0
    for line in f2:
        count += 1
        if count > num:
            break
        texts.append(line)
        labels.append('neg')
    f2.close()

    return texts, labels


def build_vocab(train_path, vocab_path, vocab_size):
    f = open(train_path, 'r', encoding='utf-8')
    f1 = open(vocab_path, 'w', encoding='utf-8')
    vocab = {}
    for line in f:
        if len(line) > 0:
            tokens = word_tokenize(line)
            for token in tokens:
                if token in vocab:
                    vocab[token] += 1
                else:
                    vocab[token] = 0

    vocab = sorted(vocab.items(), key = lambda vocab:vocab[1], reverse=True)
    # print(vocab)
    for i in range(vocab_size):
        f1.write(vocab[i][0] + '\t' + str(judge_sentiment(vocab[i][0])) + '\n')
        #f1.write(vocab[i][0]+'\n')
    f.close()
    f1.close()


def prepare_vocab(vocab_path):
    word_to_id = {}
    with open(vocab_path, encoding='utf-8') as f:
        words = [_.strip().split() for _ in f.readlines()]
        for i in range(len(words)):
            word_to_id[words[i][0]] = [i, float(words[i][1])]
#           word_senti[words[i][0]] = words[i][1]
    print(words)
    print(word_to_id)
    f.close()
    return words, word_to_id


def prepare_cat():
    categories = ['pos', 'neg']
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


def id_to_cat(id):
    categories = ['pos', 'neg']
    return categories[id]


def prepare_data(pos_path, neg_path, word_to_id, cat_to_id, max_length, num):
    contents, labels = read_file(pos_path, neg_path, num)
    data_id = []
    label_id = []

    pos = []

    for i in range(len(contents)):
        content = word_tokenize(contents[i])

        pos.append([i for i in range(len(content))])

        data_id.append([word_to_id[x][0] for x in content if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    pos_pad = kr.preprocessing.sequence.pad_sequences(pos, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=2)
    #print(x_pad[1:5])
    return x_pad, y_pad, pos_pad


def prepare_senti_data(pos_path, neg_path, word_to_id, cat_to_id, max_length, num):
    contents, labels = read_file(pos_path, neg_path, num)
    data_id = []
    label_id = []
    pos = []

    for i in range(len(contents)):
        content = word_tokenize(contents[i])

        pos.append([i for i in range(len(content))])
        #print(pos)
        data_id.append([word_to_id[x][0]*word_to_id[x][1]/2 for x in content if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    #print(pos)
    #print(data_id)
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    pos_pad = kr.preprocessing.sequence.pad_sequences(pos, max_length)

    y_pad = kr.utils.to_categorical(label_id, num_classes=2)
    # print(x_pad[1:5])
    print(x_pad)

    return x_pad, y_pad, pos_pad


#build_vocab(POS_PATH, VOCAB_PATH, vocab_size= 5000)
# read_file(POS_PATH, NEG_PATH)
#words, word_to_id = prepare_vocab(VOCAB_PATH)
# cat, cat_to_id = prepare_cat()
# prepare_data(POS_PATH, NEG_PATH, word_to_id, cat_to_id, 200)
# prepare_senti_data(POS_PATH, NEG_PATH, cat_to_id, 200)