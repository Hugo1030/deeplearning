import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import jieba
import time


# load file and get data
def load_data(file):
    with open(file) as f:
        data = []
        labels = []
        for line in f.readlines():
            words = []
            for word in jieba.cut(line.strip().split('\t')[0]):
                if word>=u'\u4e00' and word<=u'\u9fff':
                    words.append(word)
            data.append(words)
            labels.append(line.strip().split('\t')[1])
        return data,labels

# filename
train_file = "corpus/train_shuffle.txt"
test_file = "corpus/test_shuffle.txt"

train_data,train_labels = load_data(train_file)
test_data,test_labels = load_data(test_file)

# fixlenth and padding
padding = ["<Pad>"]
fix_length = 20

def pad_data(data):
    content = []

    for sentence in data:
        if len(sentence) < fix_length:
            sentence.extend(padding * (fix_length - len(sentence)))
        else:
            sentence = sentence[0: fix_length]
        content.append(sentence)
    return content

train_content = pad_data(train_data)
test_content = pad_data(test_data)

# statistical word frequency
def word_count(content):
    all_words = []
    for words in content:
        for word in words:
            all_words.append(word)
    word_cnt = Counter(all_words)
    return word_cnt
word_cnt = word_count(train_content)

# create vocab
vocab = ["UNK"]
for word,num in word_cnt.most_common():
    if num > 3:
        vocab.append(word)

# create index
idx_dict = dict(zip(vocab, range(len(vocab))))

# get sentence squency
def index_list(content):
    all_index = []
    for sentence in content:
        idx_sentence = []
        for word in sentence:
            idx = idx_dict[word] if word in idx_dict else 0
            idx_sentence.append(idx)
        all_index.append(idx_sentence)
    return all_index
train_inputs = index_list(train_content)
test_inputs = index_list(test_content)

# lst to array
x_train = np.asarray(train_inputs).astype("int32")
y_train = np.asarray(train_labels).astype("int32")

x_test = np.asarray(test_inputs).astype("int32")
y_test = np.asarray(test_labels).astype("int32")

print(x_train.shape, x_test.shape, len(vocab))

# reset tensorflow_graph
tf.reset_default_graph()
word_embedding_dim = 16
vocab_size = len(vocab)

# input layer
inputs_data = tf.placeholder(tf.int32, shape=[None, fix_length], name="inputs_data")
labels = tf.placeholder(tf.int32, shape=[None], name="labels")
word_embedding = tf.Variable(tf.random_uniform([vocab_size, word_embedding_dim]))

word_embeds = tf.nn.embedding_lookup(word_embedding, inputs_data)
embeds_reduced = tf.reduce_sum(word_embeds, axis=1)

# hidden layer
hidden = tf.layers.dense(embeds_reduced, word_embedding_dim, activation=tf.tanh)

# output layer
logits = tf.layers.dense(hidden, 2)
final_output = tf.nn.softmax(logits)

# loss function
cost = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels))
