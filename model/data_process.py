import jieba
import codecs
import re
import os
import pickle
from config import *
import numpy as np
from gensim.models import word2vec
from sklearn.decomposition import PCA
import logging
import random
import warnings
warnings.filterwarnings("ignore")

# 数据处理模块
UTIL_CN_NUM = {
    '0': '零', '1': '一', '2': '二', '两': '二', '3': '三', '4': '四',
    '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
    '壹': '一', '贰': '二', '叁': '三', '肆': '四',
    '伍': '五', '陆': '六', '柒': '七', '捌': '八', '玖': '九'}

UTIL_CN_UNIT = {2: '十', 3: '百', 4: '千', 5: '万', 6: '十万',
                7: '百万', 8: '千万', 9: '亿',
                '拾': '十', '佰': '百', '仟': '千', '萬': '万'}


# 加载词向量
def load_vec(file):
    vec = pickle.load(open(file, "rb"), encoding="UTF-8")
    logging.info("Loaded word2vec")
    return vec


# 数字转换为文字
def num_trans(nums):
    if len(nums) == 1 and nums[0] == '0':
        return '零'
    tmp = []
    for j in range(len(nums)):
        if nums[j] != '0' or (tmp and tmp[-1] != '零'):
            tmp.append(UTIL_CN_NUM[nums[j]])
            if tmp[-1] != '零' and len(nums) - j in UTIL_CN_UNIT:
                tmp.append(UTIL_CN_UNIT[len(nums) - j])
    res = ''.join(tmp)
    if len(res) > 1 and res[-1] == '零':
        res = res[:-1]
    return res


# 一句中数字转文字
def nums2cn(line):
    if line == "":
        return line
    nums = re.findall(r'\d+%|%\d+|\d+', line)
    if len(nums) == 0:
        return line
    for i in range(len(nums)):
        if '%' not in nums[i]:
            line = line.replace(nums[i], num_trans(nums[i]))
        else:
            tmp = nums[i].strip("%")
            line = line.replace(nums[i], '百分之' + num_trans(tmp))
    return line


# 词频统计去除低频词
def count_word(mat, button=10):
    word_dic = {}
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            if mat[i][j] in word_dic:
                word_dic[mat[i][j]] += 1
            else:
                word_dic[mat[i][j]] = 1
    word_lst = sorted(word_dic, key=word_dic.__getitem__)

    for m in range(len(mat)):
        for n in range(len(mat[m])):
            for k in range(button):
                if n < len(mat[m]) and word_lst[k] == mat[m][n]:
                    mat[m].pop(n)
    return mat


# 中文n-gram模型
def _word_ngrams(tokens, stop_words=None, ngram_range=(1, 1)):
    """Turn tokens into a sequence of n-grams after stop words filtering"""
    # handle stop words
    if stop_words is not None:
        tokens = [w for w in tokens if w not in stop_words]

    # handle token n-grams
    min_n, max_n = ngram_range
    if max_n != 1:
        original_tokens = tokens
        tokens = []
        n_original_tokens = len(original_tokens)
        for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
            for i in range(n_original_tokens - n + 1):
                tokens.append("".join(original_tokens[i: i + n]))
    return tokens


# 删除无用符号与词语，并分词
def pre_process(file, kind='classify', svm=False):
    data = []
    label = []
    index = []
    ner = []
    stoplist = [line.strip() for line in codecs.open(stopwords_file, encoding='utf-8')]
    r = u'[a-zA-Z’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    for line in open(file, "r", encoding="utf-8").readlines():
        line = line.strip().split(":")
        if kind == 'classify':
            # 去除无用符号
            sentence = re.sub(r, '', line[-1])
            label.append(line[-2])
            index.append(line[1])
            sentence = nums2cn(sentence)
            # n-gram + 去停用词
            # sentence = _word_ngrams(list(sentence), stop_words=stoplist, ngram_range=(1, 2))
            if svm:
                sentence = ' '.join([e for e in jieba.cut(''.join(sentence))])
                data.append(sentence)
            else:
                sentence = jieba.cut(sentence, cut_all=False)
                data.append(list(sentence))
        elif kind == 'mul_ner':
            sentence = re.sub(r, '', line[-2])
            label.append(line[-3])
            index.append(line[1])
            ner.append(line[-1].split(' '))
            sentence = nums2cn(sentence).strip().split(' ')
            # n-gram + 去停用词
            # sentence = _word_ngrams(sentence, stop_words=stoplist, ngram_range=(1, 2))
            data.append(list(sentence))
        elif kind == 'ner':
            index.append(line[1])
            ner.append(line[-2].split(' '))
            sentence = line[-3]
            # n-gram + 去停用词
            # sentence = _word_ngrams(sentence, stop_words=stoplist, ngram_range=(1, 2))
            data.append(list(sentence))
    if kind == 'classify':
        return data, label, index, None
    elif kind == 'mul_ner':
        return data, label, index, ner
    elif kind == 'ner':
        return data, None, index, ner


# 预训练word2vec
def pre_wordmodel(data, word_size):
    if not os.path.exists(word2vec_model_file):
        model = word2vec.Word2Vec(data, sg=1, size=word_size)
        model.save(word2vec_model_file)
        model.wv.save_word2vec_format(word2vec_model_txt_file, word2vec_vocab_txt_file, binary=False)
    else:
        logging.info("Word2vec model exists")
        model = word2vec.Word2Vec.load(word2vec_model_file)
    return model


# 构建嵌入字典
def build_dic(data, index, word_size, kind='normal'):
    embedding_dic = {}
    a = {}
    for word, i in index.items():
        if word in data:
            a[i] = data[word][0: word_size]
            embedding_dic[i] = data[word][0: word_size]
    if kind == 'normal':
        return embedding_dic
    else:
        pca = PCA(n_components=50)
        embedding_lst = pca.fit_transform(list(embedding_dic.values()))
        for i in range(len(embedding_dic)):
            embedding_dic[i] = embedding_lst[i]
        return embedding_dic


# 创建嵌入矩阵
def build_matrix(data, index, word_size):
    em = np.zeros((len(index) + 1, word_size))
    for word, i in index.items():
        if word in data:
            em[i] = data[word][0: word_size]
    return em


# 直接拼接句向量
def words_joint(seq, emb_dic):
    for i in range(len(seq)):
        tmp_lst = []
        for j in range(len(seq[i])):
            if seq[i][j] in emb_dic:
                tmp_lst += list(emb_dic[seq[i][j]])
        seq[i] = tmp_lst
    return seq


# 平均句向量
def words_ave(seq, emb_dic, word_size):
    for i in range(len(seq)):
        tmp_arr = np.zeros(word_size)
        for j in range(len(seq[i])):
            if seq[i][j] in emb_dic:
                tmp_arr += emb_dic[seq[i][j]]
        if len(seq[i]) > 0:
            seq[i] = np.divide(tmp_arr, len(seq[i]))
    return seq


# 最大句向量
def words_max(seq, emb_dic, word_size):
    for i in range(len(seq)):
        tmp_arr = np.zeros(word_size)
        for j in range(len(seq[i])):
            if seq[i][j] in emb_dic:
                tmp_arr = max(list(tmp_arr), list(emb_dic[seq[i][j]][:word_size]))
        if len(seq[i]) > 0:
            seq[i] = tmp_arr
    return seq


# 标签转数字
def label2num(train, test):
    dic_label = {}
    train = list(map(int, train))
    test = list(map(int, test))
    label = sorted(set(train))
    j = 0
    for i in label:
        dic_label[i] = j
        j += 1
    for m in range(len(train)):
        if m < len(test):
            test[m] = dic_label[test[m]]
        train[m] = dic_label[train[m]]
    return train, test


# 命名转数字
def ner2num(data, dic):
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = dic[data[i][j]]
    return data


# 存txt文件
def save_txt(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for i in data:
            f.writelines(i)
            
            
# 划分数据集
def divide_data(file_in, file_out_train, file_out_test, shuffle=True, train_rate=0.9):
    with open(file_in, 'r', encoding='utf-8') as txtData:
        lines = txtData.readlines()
        index = list(range(len(lines)))
        if shuffle:
            random.shuffle(index)
            train, test = [], []
            for i in range(len(index)):
                if i in index[0: int(len(index)*train_rate)]:
                    train.append(lines[i])
                else:
                    test.append(lines[i])
        else:
            train = lines[0: int(len(lines)*train_rate)]
            test = lines[int(len(lines)*train_rate):]
        print('train_size:{} test_size:{}'.format(len(train), len(test)))
        save_txt(file_out_train, train)
        save_txt(file_out_test, test)
