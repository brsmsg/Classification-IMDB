from gensim.models import Word2Vec
from dataHelper import read_file
from nltk.tokenize import word_tokenize

POS_PATH = 'data\\train\\pos_all.txt'
NEG_PATH = 'data\\train\\neg_all.txt'
TEST_POS_PATH = 'data\\test\\pos_all.txt'
TEST_NEG_PATH = 'data\\test\\neg_all.txt'


def train_vec():
    contents, label = read_file(POS_PATH, NEG_PATH, 12500)
    contents2, label = read_file(TEST_POS_PATH, TEST_NEG_PATH, 12500)
    texts = []
    for content in contents:
        text = word_tokenize(content)
        texts.append(text)

    for content in contents2:
        text = word_tokenize(content)
        texts.append(text)

    model = Word2Vec(texts, min_count=2)
    model.save('data/w2vmodel')


#train_vec()
model = Word2Vec.load('data/w2vmodel')

print(model.wv.__getitem__('movie'))
