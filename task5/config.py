# -*- encoding: utf-8 -*-


class Config(object):
    data_path = "./json/"
    category = "poet.tang"
    author = None
    constrain = None
    poetry_max_len = 125
    sample_max_len = poetry_max_len-1
    processed_data_path = "data/tang.npz"
    word_dict_path = 'wordDic'
    model_path = 'model/tang_200.pth'
    model_prefix = 'model/tang'

    batch_size = 128
    epoch_num = 201

    embedding_dim = 256
    hidden_dim = 256
    layer_num = 2  # rnn的层数
    lr = 0.01
    weight_decay = 1e-4

    plot_every = 2
    debug_file = '/tmp/debugp'
    env = 'poetry'

    use_gpu = False

    max_gen_len = 200  # 生成诗歌最长长度
    sentence_max_len = 4 # 生成诗歌的最长句子

    prefix_words = '细雨鱼儿出,微风燕子斜。'  # 不是诗歌的组成部分，用来控制生成诗歌的意境
    start_words = '闲云潭影日悠悠'  # 诗歌开始
    acrostic = False  # 是否是藏头诗







