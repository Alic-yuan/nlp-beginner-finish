# coding: utf-8

from __future__ import print_function

import os
import tensorflow.contrib.keras as kr
import torch
from torch import nn
from cnews_loader import read_category, read_vocab
from torch_model import TextCNN,TextRNN
from torch.autograd import Variable
import numpy as np

try:
    bool(type(unicode))
except NameError:
    unicode = str

base_dir = 'cnews'
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')


class CnnModel:
    def __init__(self):
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.model = TextCNN()
        self.model.load_state_dict(torch.load('model_params.pkl'))

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]
        data = kr.preprocessing.sequence.pad_sequences([data], 600)
        data = torch.LongTensor(data)
        y_pred_cls = self.model(data)
        print(y_pred_cls)
        class_index = torch.argmax(y_pred_cls[0]).item()
        return self.categories[class_index]


class RnnModel:
    def __init__(self):
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.model = TextRNN()
        self.model.load_state_dict(torch.load('model_params.pkl'))

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]
        data = kr.preprocessing.sequence.pad_sequences([data], 600)
        data = torch.LongTensor(data)
        y_pred_cls = self.model(data)
        class_index = torch.argmax(y_pred_cls[0]).item()
        return self.categories[class_index]



if __name__ == '__main__':
    # model = CnnModel()
    model = RnnModel()
    test_demo = ['三星ST550以全新的拍摄方式超越了以往任何一款数码相机',
                 '热火vs骑士前瞻：皇帝回乡二番战 东部次席唾手可得新浪体育讯北京时间3月30日7:00']
    for i in test_demo:
        print(i,":",model.predict(i))


