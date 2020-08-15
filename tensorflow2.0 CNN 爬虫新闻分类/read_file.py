# encoding: UTF-8

import numpy as np
import re
import itertools
from collections import Counter
import os
#import word2vec_helpers
import time
import pickle
from xlrd import open_workbook  # xlrd用于读取xld
import xlwt  # 用于写入xls
import jieba
import re
import  time
import  numpy as np
import pandas as pd
from tqdm import tqdm
import  tensorflow as tf

def load_positive_negative_data_files(keji_data_file, nba2_data_file,nba_data_file):

    # 读取文件
    keji_examples = read_and_clean_zh_file(keji_data_file)
    nba2_examples = read_and_clean_zh_file(nba2_data_file)
    nba_examples  = read_and_clean_zh_file(nba_data_file)
    #划分训练集 测试集
    x_train = keji_examples[:1000] + nba_examples[:600] + nba2_examples[:600]
    x_text  = keji_examples[1000:] + nba_examples[600:] + nba2_examples[600:]
    #给新闻做标签
    keji_labels = [[0] for _ in keji_examples[:1000]]
    nba_labels = [[1] for _ in nba_examples[:600]]
    nba2_labels = [[1] for _ in nba2_examples[:600]]
    y_train = np.concatenate([keji_labels, nba_labels,nba2_labels], 0)

    keji_ = [[0] for _ in keji_examples[1000:]]
    nba_ = [[1] for _ in nba_examples[600:]]
    nba2_ = [[1] for _ in nba2_examples[600:]]
    y_test = np.concatenate([keji_, nba_,nba2_], 0)
    x_train = clen_stopword(x_train)
    x_text=clen_stopword(x_text)

    return [x_train,y_train,x_text, y_test]


def read_and_clean_zh_file(input_file, output_cleaned_file=None):
    # lines = list(open(input_file, "rb").readlines())
    workbook = open_workbook(input_file)  # 打开xls文件\
    sheet = workbook.sheet_by_index(0)
    content = sheet.row_values(4)  # 第4列内容
    lines = content[1:]
    lines = seperate_line(lines)

    return lines

def seperate_line(lines):
    li = []
    data = []
    for i in range(len(lines)):
       if lines[i] !='' :
         reg = "[^\u4e00-\u9fa5]"
         li.append(re.sub(reg,'',lines[i]))
    for e in range(len(li)):
        data.append(jieba.lcut(li[e]))
    return data

def clen_stopword(x_text):

    content = []
    stopwords = pd.read_csv("stopwords.txt", index_col=False, sep="\t", quoting=3, names=['stopword'], encoding='utf-8')
    stopwords = stopwords.stopword.values.tolist()
    for i in tqdm(x_text):
        #print(i)
        s = []
        for n in i :
            if n not  in stopwords:
                s.append(n)
       # print(s)
        content.append(s)
    return  content


def padding_sentences(input_sentences, padding_token, padding_sentence_length = None):
    max_sentence_length = 4592
    sentences = []
    for sentence in input_sentences :
        if len(sentence) > max_sentence_length:
            sentence = sentence[:max_sentence_length]
            sentences.append(sentence)
        else:
            sentence.extend([padding_token] * (max_sentence_length - len(sentence)))
            sentences.append(sentence)
    return (sentences, max_sentence_length)




if __name__ == '__main__':
    keji_data_file="./data/keji.xlsx"
    nba_data_file="./data/NBA.xlsx"
    nba2_data_file = './data/NBA2.xlsx'
    x_train,y_train,x_text, y_test = load_positive_negative_data_files(keji_data_file, nba2_data_file,nba_data_file)
    #sentences, max_document_length = padding_sentences(cont, '<PADDING>')
    x_train= np.array(x_train)
    y_train = np.array(y_train)
    print(x_train[1])
    y_train = tf.squeeze(y_train, axis=1)
    print(x_train.shape)
    print(y_train.shape)






