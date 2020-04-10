#!/usr/bin/env python

import os
import json
import re
import numpy as np


def parse_raw_data(data_path, category, author, constrain):
    """
    获取原数据并预处理
    :param data_path: 数据存放的路径
    :param category: 数据的类型
    :param author: 作者名称
    :param constrain: 长度限制
    :return: list
    ['床前明月光，疑是地上霜，举头望明月，低头思故乡。',
     '一去二三里，烟村四五家，亭台六七座，八九十支花。',
    .........
    ]
    """
    def sentence_parse(para):
        """对文本进行处理，取出脏数据"""
        # 去掉括号中的部分
        # para = "-181-村橋路不端，數里就迴湍。積壤連涇脉，高林上笋竿。早嘗甘蔗淡，生摘琵琶酸。（「琵琶」，嚴壽澄校《張祜詩集》云：疑「枇杷」之誤。）好是去塵俗，煙花長一欄。"
        result, number = re.subn("（.*）", "", para)
        result, number = re.subn("{.*}", "", result)
        result, number = re.subn("《.*》", "", result)
        result, number = re.subn("《.*》", "", result)
        result, number = re.subn("[\]\[]", "", result)
        # 去掉数字
        r = ""
        for s in result:
            if s not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']:
                r += s;
        # 处理两个句号为1个句号
        r, number = re.subn("。。", "。", r)

        # 返回预处理好的文本
        return r

    def handle_json(file):
        """读入json文件，返回诗句list，每一个元素为一首诗歌(str类型表示)"""
        rst = []
        data = json.loads(open(file).read())
        for poetry in data:
            pdata = ""
            if author is not None and poetry.get("author") != author:
                continue
            p = poetry.get("paragraphs")
            flag = False
            for s in p:
                sp = re.split("[，！。]", s)
                for tr in sp:
                    if constrain is not None and len(tr) != constrain and len(tr) != 0:
                        flag = True
                        break
                    if flag:
                        break
            if flag:
                continue

            for sentence in poetry.get("paragraphs"):
                pdata += sentence
            pdata = sentence_parse(pdata)
            if pdata != "" and len(pdata) > 1:
                rst.append(pdata)
        return rst

    data = []
    for filename in os.listdir(data_path):
        if filename.startswith(category):
            data.extend(handle_json(data_path + filename))
    return data


def pad_sequences(sequences,
                  maxlen=None,
                  dtype='int32',
                  padding='pre',
                  truncating='pre',
                  value=0.):
    """
    code from keras
    Pads each sequence to the same length (length of the longest sequence).
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    Arguments:
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    Returns:
        x: numpy array with dimensions (number_of_sequences, maxlen)
    Raises:
        ValueError: in case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:  # pylint: disable=g-explicit-length-test
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):  # pylint: disable=g-explicit-length-test
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]  # pylint: disable=invalid-unary-operand-type
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                'Shape of sample %s of sequence at position %s is different from '
                'expected shape %s'
                % (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def get_data(config):
    # 1.获取数据
    data = parse_raw_data(config.data_path, config.category, config.author, config.constrain)

    # 2.构建词典
    chars = {c for line in data for c in line}
    char_to_ix = {char: ix for ix, char in enumerate(chars)}
    char_to_ix['<EOP>'] = len(char_to_ix)
    char_to_ix['<START>'] = len(char_to_ix)
    char_to_ix['</s>'] = len(char_to_ix)

    ix_to_chars = {ix: char for char, ix in list(char_to_ix.items())}

    # 3.处理样本
    # 3.1 每首诗加上首位符号
    for i in range(0, len(data)):
        data[i] = ['<START>'] + list(data[i]) + ['<EOP>']

    # 3.2 文字转id
    data_id = [[char_to_ix[w] for w in line] for line in data]

    # 3.3 补全既定长度
    pad_data = pad_sequences(data_id,
                             maxlen=config.poetry_max_len,
                             padding='pre',
                             truncating='post',
                             value=len(char_to_ix) - 1)

    # 3.4 保存于返回
    np.savez_compressed(config.processed_data_path,
                        data=pad_data,
                        word2ix=char_to_ix,
                        ix2word=ix_to_chars)

    return pad_data, char_to_ix, ix_to_chars


if __name__ == '__main__':
    from config import Config
    config = Config()
    pad_data, char_to_ix, ix_to_chars = get_data(config)
    for l in pad_data[:10]:
        print(l)

    n = 0
    for k, v in char_to_ix.items():
        print(k, v)
        if n > 10:
            break
        n += 1

    n = 0
    for k, v in ix_to_chars.items():
        print(k, v)
        if n > 10:
            break
        n += 1





