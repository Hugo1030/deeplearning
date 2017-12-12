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

# cnn init
word_embed_size = 128
filter_num = 64
window_size = 3
vocab_size = len(vocab)

# inputs_layer
tf.reset_default_graph()
filter_shape = [window_size, word_embed_size, 1, filter_num]
word_embedding = tf.Variable(tf.random_uniform([vocab_size, word_embed_size], -1.0, 1.0), name='word_embedding')
x_ = tf.placeholder(tf.int32, shape=[None, fix_length], name="x_")
y_ = tf.placeholder(tf.int32, shape=[None], name="y_")
word_embeds = tf.nn.embedding_lookup(word_embedding, x_)

# expand dimension
embeds_expand = tf.expand_dims(word_embeds, -1)
print(embeds_expand)
#  conv and maxpooling layer
with tf.name_scope("conv-maxpool"):
    filter_shape = [window_size, word_embed_size, 1, filter_num]
    W = tf.Variable(tf.random_uniform(filter_shape, -1.0, 1.0), name="W")
    b = tf.Variable(tf.constant(0.0, shape=[filter_num]), name='b')

    conv = tf.nn.conv2d(
        embeds_expand, W, strides=[1, 1, 1, 1],
        padding='VALID', name='conv')

    conv_hidden = tf.nn.tanh(tf.add(conv, b), name='tanh')

    pool = tf.nn.max_pool(
        conv_hidden,
        ksize=[1, fix_length - window_size + 1, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID',
        name='pool')

squeezed_pool = tf.squeeze(pool, [1, 2])

# output layer
raw_output = tf.layers.dense(squeezed_pool, 2)
output = tf.nn.softmax(raw_output)

# init cost
cost = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=raw_output, labels=y_))

# evaluate model
def evaluate_model(sess, inputs_, labels_):
    pred_prob = sess.run(output, feed_dict={x_: inputs_, y_: labels_})
    preds = np.asarray((pred_prob[:, 1]>0.5), dtype=int)
    mat = sess.run(tf.confusion_matrix(labels_, preds))
    tn, fp, fn, tp = mat.reshape(4)
    Accuracy = (tp + tn) / (tn + tp + fn + fp)
    return Accuracy,mat

train_step = tf.train.AdamOptimizer(3e-4).minimize(cost)

#lm init
sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_dict = {x_: x_train, y_: y_train}
test_dict = {x_: x_test, y_: y_test}

costs_train = []
costs_test = []

arr = np.arange(len(x_train))
batch_size = 50

# train_model
start = time.time()
try:
    for i in range(10000):
        np.random.shuffle(arr)
        for j in range(batch_size):
            batch_index = arr[j: j+batch_size]
            batch_inputs = x_train[batch_index]
            batch_labels = y_train[batch_index]
            batch_dict = {x_: batch_inputs, y_: batch_labels}
            sess.run(train_step, feed_dict=batch_dict)

        if i % 10 == 0:
            cost_train = sess.run(cost, feed_dict=train_dict)
            cost_test = sess.run(cost, feed_dict=test_dict)
            accuracy_train,mat_train = evaluate_model(sess, x_train, y_train)
            accuracy_test,mat_test = evaluate_model(sess, x_test, y_test)
            print("Epoch {:03d} cost: train {:.3f} / test {:.3f}".format(
                i, cost_train, cost_test))
            print("Accuracy {:.3f}: trian / test {:.3f}".format(accuracy_train, accuracy_test))
except KeyboardInterrupt:
     print("KeyboardInterrupt.")
finally:
    accuracy,mat = evaluate_model(sess, x_test, y_test)
    end = time.time()
    print("\ntime: {:.2f} s".format(end - start))
    print(" Confusion matrix:\n", mat)
