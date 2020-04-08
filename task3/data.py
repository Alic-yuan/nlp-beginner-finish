import re
import gensim
import numpy as np
import pandas as pd
import torch
from hanziconv import HanziConv
from torch.utils.data import Dataset
from collections import Counter



class LCQMC_Dataset(Dataset):
    def __init__(self, LCQMC_file, vocab_file, max_char_len):
        p, h, self.label = load_sentences(LCQMC_file)
        word2idx, _, _ = load_vocab(vocab_file)
        self.p_list, self.p_lengths, self.h_list, self.h_lengths = word_index(p, h, word2idx, max_char_len)
        self.p_list = torch.from_numpy(self.p_list).type(torch.long)
        self.h_list = torch.from_numpy(self.h_list).type(torch.long)
        self.max_length = max_char_len

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.p_list[idx], self.p_lengths[idx], self.h_list[idx], self.h_lengths[idx], self.label[idx]

def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    return open(filename, mode, encoding='utf-8', errors='ignore')


# 加载word_index训练数据
def load_sentences(file, data_size=None):
    df = pd.read_csv(file, sep='\t', names=['sentence1', 'sentence2', 'label'])
    # print(df)
    p = map(get_word_list, df['sentence1'].values[0:data_size])
    h = map(get_word_list, df['sentence2'].values[0:data_size])
    label = df['label'].values[0:data_size]
    # # p_c_index, h_c_index = word_index(p, h)
    return p, h, label

# word->index
def word_index(p_sentences, h_sentences, word2idx, max_char_len):
    p_list, p_length, h_list, h_length = [], [], [], []
    for p_sentence, h_sentence in zip(p_sentences, h_sentences):
        p = [word2idx[word] for word in p_sentence if word in word2idx.keys()]
        h = [word2idx[word] for word in h_sentence if word in word2idx.keys()]
        p_list.append(p)
        p_length.append(min(len(p), max_char_len))
        h_list.append(h)
        h_length.append(min(len(h), max_char_len))
    p_list = pad_sequences(p_list, maxlen = max_char_len)
    h_list = pad_sequences(h_list, maxlen = max_char_len)
    return p_list, p_length, h_list, h_length

# 加载字典
def load_vocab(vocab_file):
    vocab = [line.strip() for line in open(vocab_file, encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word, vocab

''' 把句子按字分开，中文按字分，英文数字按空格, 大写转小写，繁体转简体'''
def get_word_list(query):
    query = HanziConv.toSimplified(query.strip())
    regEx = re.compile('[\\W]+')#我们可以使用正则表达式来切分句子，切分的规则是除单词，数字外的任意字符串
    res = re.compile(r'([\u4e00-\u9fa5])')#[\u4e00-\u9fa5]中文范围
    sentences = regEx.split(query.lower())
    str_list = []
    for sentence in sentences:
        if res.split(sentence) == None:
            str_list.append(sentence)
        else:
            ret = res.split(sentence)
            str_list.extend(ret)
    return [w for w in str_list if len(w.strip()) > 0]

def load_embeddings(embdding_path):
    model = gensim.models.KeyedVectors.load_word2vec_format(embdding_path, binary=False)
    embedding_matrix = np.zeros((len(model.index2word) + 1, model.vector_size))
    #填充向量矩阵
    for idx, word in enumerate(model.index2word):
        embedding_matrix[idx + 1] = model[word]#词向量矩阵
    return embedding_matrix

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):
    """ pad_sequences
    把序列长度转变为一样长的，如果设置了maxlen则长度统一为maxlen，如果没有设置则默认取
    最大的长度。填充和截取包括两种方法，post与pre，post指从尾部开始处理，pre指从头部
    开始处理，默认都是从尾部开始。
    Arguments:
        sequences: 序列
        maxlen: int 最大长度
        dtype: 转变后的数据类型
        padding: 填充方法'pre' or 'post'
        truncating: 截取方法'pre' or 'post'
        value: float 填充的值
    Returns:
        x: numpy array 填充后的序列维度为 (number_of_sequences, maxlen)
    """
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    # print(maxlen)
    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)
        # x[idx] = trunc
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x

def build_vocab(train_dir, vocab_size=5000):
    df = pd.read_csv(train_dir, sep='\t', names=['sentence1', 'sentence2', 'label'])
    # print(df)
    p = map(get_word_list, df['sentence1'].values)
    h = map(get_word_list, df['sentence2'].values)
    p2 = ''.join(''.join(i) for i in p)
    h2= ''.join(''.join(i) for i in h)
    text = p2 + h2
    counter = Counter(text)
    # print(counter)
    count_pairs = counter.most_common(vocab_size - 1)  #统计最常出现的字
    print(count_pairs)
    words, _ = list(zip(*count_pairs))
    print(words)
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file('vocab.txt', mode='w').write('\n'.join(words) + '\n')

if __name__ == '__main__':
    build_vocab('atec_nlp_sim_train_all.csv')