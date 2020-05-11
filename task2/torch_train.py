# coding: utf-8

from __future__ import print_function
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import os

import numpy as np

from torch_model import TextRNN, TextCNN
from cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab

def evaluate(model, Loss, x_val, y_val):
    """测试集上准确率评估"""
    batch_val = batch_iter(x_val, y_val, 64)
    acc = 0
    los = 0
    for x_batch, y_batch in batch_val:
        size = len(x_batch)
        x = np.array(x_batch)
        y = np.array(y_batch)
        x = torch.LongTensor(x)
        y = torch.Tensor(y)
        # y = torch.LongTensor(y)
        # x = Variable(x)
        # y = Variable(y)
        out = model(x)
        loss = Loss(out, y)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        loss_value = np.mean(loss.detach().numpy())
        accracy = np.mean((torch.argmax(out, 1) == torch.argmax(y, 1)).numpy())
        acc +=accracy*size
        los +=loss_value*size
    return los/len(x_val), acc/len(x_val)


base_dir = 'cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

def train():
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id,600)#获取训练数据每个字的id和对应标签的oe-hot形式
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id,600)
    #使用LSTM或者CNN
    model = TextRNN()
    # model = TextCNN()
    #选择损失函数
    Loss = nn.MultiLabelSoftMarginLoss()
    # Loss = nn.BCELoss()
    # Loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    best_val_acc = 0
    for epoch in range(100):
        i = 0
        print('epoch:{}'.format(epoch))
        batch_train = batch_iter(x_train, y_train,64)
        for x_batch, y_batch in batch_train:
            i +=1
            # print(i)
            x = np.array(x_batch)
            y = np.array(y_batch)
            x = torch.LongTensor(x)
            y = torch.Tensor(y)
            # y = torch.LongTensor(y)
            # x = Variable(x)
            # y = Variable(y)
            out = model(x)
            loss = Loss(out,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 对模型进行验证
            if i % 90 == 0:
                los, accracy = evaluate(model, Loss, optimizer, x_val, y_val)
                print('loss:{},accracy:{}'.format(los, accracy))
                if accracy > best_val_acc:
                    torch.save(model.state_dict(), 'model_params.pkl')
                    best_val_acc = accracy


if __name__ == '__main__':
    #获取文本的类别及其对应id的字典
    categories, cat_to_id = read_category()
    #获取训练文本中所有出现过的字及其所对应的id
    words, word_to_id = read_vocab(vocab_dir)
    #获取字数
    vocab_size = len(words)
    print('train')
    train()



