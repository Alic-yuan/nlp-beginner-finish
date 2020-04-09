import pickle
import pdb
import numpy as np 
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from config import opt 
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.optim as optim
from model.LSTM import NERLSTM
from model.LSTM_CRF import NERLSTM_CRF
from utils import get_tags, format_result


with open(opt.pickle_path, 'rb') as inp:
        word2id = pickle.load(inp)
        # id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        # id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)
        x_valid = pickle.load(inp)
        y_valid = pickle.load(inp)
print("train len:",len(x_train))
print("test len:",len(x_test))
print("valid len", len(x_valid))
print(word2id)
print(tag2id)


class NERDataset(Dataset):
    def __init__(self, X, Y, *args, **kwargs):
        self.data = [{'x':X[i],'y':Y[i]} for i in range(X.shape[0])]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

train_dataset = NERDataset(x_train, y_train)
valid_dataset = NERDataset(x_valid, y_valid)
test_dataset = NERDataset(x_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)


models = {'NERLSTM': NERLSTM,
          'NERLSTM_CRF': NERLSTM_CRF}

model = models[opt.model](opt.embedding_dim, opt.hidden_dim, opt.dropout, word2id, tag2id)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)




class ChineseNER(object):
    def train(self):
        for epoch in range(opt.max_epoch):
            model.train()
            for index, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                X = batch['x']
                y = batch['y']
                #CRF
                loss = model.log_likelihood(X,y)
                loss.backward()
                #CRF
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)

                optimizer.step()
                if index % 200 == 0:
                    print('epoch:%04d,------------loss:%f'%(epoch,loss.item()))

            aver_loss = 0
            preds, labels = [], []
            for index, batch in enumerate(valid_dataloader):
                model.eval()
                val_x,val_y = batch['x'], batch['y']
                predict = model(val_x)
                #CRF
                loss = model.log_likelihood(val_x, val_y)
                aver_loss += loss.item()
                # 统计非0的，也就是真实标签的长度
                leng = []
                for i in val_y.cpu():
                    tmp = []
                    for j in i:
                        if j.item()>0:
                            tmp.append(j.item())
                    leng.append(tmp)

                for index, i in enumerate(predict):
                    preds += i[:len(leng[index])]

                for index, i in enumerate(val_y.tolist()):
                    labels += i[:len(leng[index])]
            aver_loss /= (len(valid_dataloader) * 64)
            precision = precision_score(labels, preds, average='macro')
            recall = recall_score(labels, preds, average='macro')
            f1 = f1_score(labels, preds, average='macro')
            report = classification_report(labels, preds)
            print(report)
            torch.save(model.state_dict(), './model/params.pkl')

    def predict(self, tag, input_str=""):
        model.load_state_dict(torch.load("./model/params.pkl"))
        if not input_str:
            input_str = input("请输入文本: ")
        input_vec = [word2id.get(i, 0) for i in input_str]
        # convert to tensor
        sentences = torch.tensor(input_vec).view(1, -1)
        paths = model(sentences)

        entities = []
        tags = get_tags(paths[0], tag, tag2id)
        entities += format_result(tags, input_str, tag)
        print(entities)

if __name__ == '__main__':
    cn = ChineseNER()
    cn.train()
    # cn.predict('ns')
