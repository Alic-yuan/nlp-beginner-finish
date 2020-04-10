import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, device, layer_num):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim

        # 创建embedding层
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 创建lstm层,参数是输入输出的维度
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=layer_num)
        # 创建一个线性层
        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)
        # 创建一个dropout层，训练时作用在线性层防止过拟合
        self.dropout = nn.Dropout(0.2)

        self.device = device

    def forward(self, inputs, hidden):
        seq_len, batch_size = inputs.size()
        # 将one-hot形式的input在嵌入矩阵中转换成嵌入向量，torch.Size([length, batch_size, embedding_size])
        embeds = self.embeddings(inputs)

        # 经过lstm层，该层有2个输入，当前x和t=0时的(c,a),
        # output:torch.Size([length, batch_size, hidden_idm]), 每一个step的输出
        # hidden: tuple(torch.Size([layer_num, 32, 256]) torch.Size([1, 32, 256])) # 最后一层输出的ct 和 ht, 在这里是没有用的
        output, hidden = self.lstm(embeds, hidden)

        # 经过线性层，relu激活层 先转换成（max_len*batch_size, 256)维度，再过线性层（length, vocab_size)
        output = F.relu(self.linear1(output.view(seq_len*batch_size, -1)))

        # 输出最终结果，与hidden结果
        return output, hidden

    def init_hidden(self, layer_num, batch_size):
        return (Variable(torch.zeros(layer_num, batch_size, self.hidden_dim)),
                Variable(torch.zeros(layer_num, batch_size, self.hidden_dim)))
