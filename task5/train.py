# -*- coding: utf-8 -*-
import os
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
from model import PoetryModel
from dataHandler import *
from config import Config
from tqdm import tqdm


class TrainModel(object):
    def __init__(self):

        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        self.config = Config()
        self.device = torch.device('cuda') if self.config.use_gpu else torch.device('cpu')

    def train(self, data_loader, model, optimizer, criterion, char_to_ix, ix_to_chars):
        for epoch in range(self.config.epoch_num):
            for step, x in enumerate(data_loader):
                # 1.处理数据
                # x: (batch_size,max_len) ==> (max_len, batch_size)
                x = x.long().transpose(1, 0).contiguous()
                x = x.to(self.device)
                optimizer.zero_grad()
                # input,target:  (max_len, batch_size-1)
                input_, target = x[:-1, :], x[1:, :]
                target = target.view(-1)
                # 初始化hidden为(c0, h0): ((layer_num， batch_size, hidden_dim)，(layer_num， batch_size, hidden_dim)）
                hidden = model.init_hidden(self.config.layer_num, x.size()[1])

                # 2.前向计算
                # print(input.size(), hidden[0].size(), target.size())
                output, _ = model(input_, hidden)
                loss = criterion(output, target) # output:(max_len*batch_size,vocab_size), target:(max_len*batch_size)

                # 反向计算梯度
                loss.backward()

                # 权重更新
                optimizer.step()

                if step % 200 == 0:
                    print('epoch: %d,loss: %f' % (epoch, loss.data))

            if epoch % 1 == 0:
                # 保存模型
                torch.save(model.state_dict(), '%s_%s.pth' % (self.config.model_prefix, epoch))

                # 分别以这几个字作为诗歌的第一个字，生成一首藏头诗
                word = '春江花月夜凉如水'
                gen_poetry = ''.join(self.generate_head_test(model, word, char_to_ix, ix_to_chars))
                print(gen_poetry)



    def run(self):
        # 1 获取数据
        data, char_to_ix, ix_to_chars = get_data(self.config)
        vocab_size = len(char_to_ix)
        print('样本数：%d' % len(data))
        print('词典大小： %d' % vocab_size)

        # 2 设置dataloader
        data = torch.from_numpy(data)
        data_loader = Data.DataLoader(data,
                                      batch_size=self.config.batch_size,
                                      shuffle=True,
                                      num_workers=1)

        # 3 创建模型
        model = PoetryModel(vocab_size=vocab_size,
                            embedding_dim=self.config.embedding_dim,
                            hidden_dim=self.config.hidden_dim,
                            device=self.device,
                            layer_num=self.config.layer_num)
        model.to(self.device)

        # 4 创建优化器
        optimizer = optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        # 5 创建损失函数,使用与logsoftmax的输出
        criterion = nn.CrossEntropyLoss()

        # 6.训练
        self.train(data_loader, model, optimizer, criterion, char_to_ix, ix_to_chars)

    def generate_head_test(self, model, head_sentence, word_to_ix, ix_to_word):
        """生成藏头诗"""
        poetry = []
        head_char_len = len(head_sentence)  # 要生成的句子的数量
        sentence_len = 0  # 当前句子的数量
        pre_char = '<START>'  # 前一个已经生成的字

        # 准备第一步要输入的数据
        input = (torch.Tensor([word_to_ix['<START>']]).view(1, 1).long()).to(self.device)
        hidden = model.init_hidden(self.config.layer_num, 1)

        for i in range(self.config.max_gen_len):
            # 前向计算出概率最大的当前词
            output, hidden = model(input, hidden)
            top_index = output.data[0].topk(1)[1][0].item()
            char = ix_to_word[top_index]

            # 句首的字用藏头字代替
            if pre_char in ['。', '！', '<START>']:
                if sentence_len == head_char_len:
                    break
                else:
                    char = head_sentence[sentence_len]
                    sentence_len += 1
                    input = (input.data.new([word_to_ix[char]])).view(1,1)
            else:
                input = (input.data.new([top_index])).view(1,1)

            poetry.append(char)
            pre_char = char


        return poetry


if __name__ == '__main__':
    obj = TrainModel()
    obj.run()





