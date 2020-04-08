# coding:utf8
import warnings
import torch as t

class DefaultConfig(object):
    model = 'NERLSTM_CRF'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    pickle_path = './renmindata.pkl'  # 训练集存放路径
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 32  # batch size
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    max_epoch = 10
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-5  # 损失函数

    embedding_dim = 100
    hidden_dim = 200
    dropout = 0.2


    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        
        opt.device =t.device('cuda') if opt.use_gpu else t.device('cpu')


        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()
