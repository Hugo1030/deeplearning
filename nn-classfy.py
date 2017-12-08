import tensorflow as tf
import pandas as pd
from collections import Counter
import jieba
import time


# load file and get data
def load_data(file):
    line_no = 0
    limits = 5
    with open(file) as f:
        data = []
        labels = []
        for line in f.readlines():
            line_no += 1
            if line_no > limits:
                break
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
def pad_data(data):
    content = []
    fix_length = 20
    padding = ["<Pad>"]

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
print(word_cnt)
