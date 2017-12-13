# -*- coding: utf-8 -*-

from collections import Counter
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
import jieba
from zhon.hanzi import punctuation

# read file
read_size = 10000
with open('corpus/xiaobo.txt') as f:
    corpus = f.read(read_size)

END_PUNC = '。？！' # end of sentence
IGNORE = '\n ""... 《》（）* \u3000 :' # ignore words

sentences = []
sentence = []
for word in jieba.cut(corpus):
    if word not in IGNORE:
        sentence.append(word)
    if word in END_PUNC:
        sentences.append(sentence)
        sentence = []

# sentence start token
sentences = [['<s>'] + sen for sen in sentences]

# get sentences length
all_length = [len(line) for line in sentences]
max_length = 51

# padding
all_sentences = []
for sen in sentences:
    if len(sen) < max_length:
        sen = sen + ["<pad>" ]*(max_length-len(sen))
        all_sentences.append(sen)

# count words
all_words = []
for words in all_sentences:
    for word in words:
        all_words.append(word)
    word_cnt = Counter(all_words)

# built vocab
vocab = ['UNK']
for i in word_cnt.most_common():
    if i[1] > 5:
        vocab.append(i[0])
    else:
        break
vocab_size = len(vocab)

idx_dict = dict(zip(vocab, range(vocab_size)))

index = []
for contents in all_sentences:
    index_content = []
    for word in contents:
        idx = idx_dict[word] if (word in vocab) else 0
        index_content.append(idx)
    index.append(index_content)

sentences_vector = np.array(index)

inputs_train = sentences_vector[:, :-1]
labels_train = sentences_vector[:, 1:]

print(inputs_train.shape, labels_train.shape)
