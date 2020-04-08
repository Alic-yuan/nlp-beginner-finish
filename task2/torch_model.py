#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


class TextRNN(nn.Module):
    """文本分类，RNN模型"""

    def __init__(self):
        super(TextRNN, self).__init__()
        # 三个待输入的数据
        self.embedding = nn.Embedding(5000, 64)  # 进行词嵌入
        self.rnn = nn.LSTM(input_size=64, hidden_size=128, bidirectional=True)
        # self.rnn = nn.GRU(input_size=64, hidden_size=128, num_layers=2, bidirectional=True)
        self.f1 = nn.Sequential(nn.Linear(256, 10),
                                nn.Softmax())

    def forward(self, x):
        x = self.embedding(x) # batch_size x text_len x embedding_size 64*600*64
        x= x.permute(1, 0, 2) # text_len x batch_size x embedding_size 600*64*64
        x, (h_n, c_n)= self.rnn(x) #x为600*64*256, h_n为2*64*128 lstm_out       Sentence_length * Batch_size * (hidden_layers * 2 [bio-direct]) h_n           （num_layers * 2） * Batch_size * hidden_layers
        final_feature_map = F.dropout(h_n, 0.8)
        feature_map = torch.cat([final_feature_map[i, :, :] for i in range(final_feature_map.shape[0])], dim=1) #64*256 Batch_size * (hidden_size * hidden_layers * 2)
        final_out = self.f1(feature_map) #64*10 batch_size * class_num
        return final_out



class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(5000, 64)
        self.conv = nn.Sequential(nn.Conv1d(in_channels=64,
                                        out_channels=256,
                                        kernel_size=5),
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=596))

        self.f1 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.embedding(x) # batch_size x text_len x embedding_size 64*600*64
        x = x.permute(0, 2, 1) #64*64*600

        x = self.conv(x)  #Conv1后64*256*596,ReLU后不变,NaxPool1d后64*256*1

        x = x.view(-1, x.size(1)) #64*256
        x = F.dropout(x, 0.8)
        x = self.f1(x)    #64*10 batch_size * class_num
        return x

if __name__ == '__main__':
    net = TextRNN()
    print(net)
