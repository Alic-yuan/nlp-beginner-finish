# -*- coding:utf8 -*-

import pdb
import os
import tensorflow as tf
import numpy as np
from cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab

def evaluate(sess, x_, y_):
    """测试集上准曲率评估"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0
    total_acc = 0
    for batch_xs, batch_ys in batch_eval:
        feed_dict = feed_data(batch_xs, batch_ys)
        batch_len = len(batch_xs)
        loss2, acc = sess.run([loss, accuracy], feed_dict=feed_dict)
        total_loss += loss2 * batch_len
        total_acc += acc * batch_len
    return total_loss/data_len, total_acc/data_len

def feed_data(x_batch, y_batch):
    feed_dict = {
        x: x_batch,
        y_: y_batch
    }
    return feed_dict


base_dir = 'cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

pwd_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

lr_save_dir = os.path.join(pwd_path, 'task1', 'lr')
lr_save_path = os.path.join(lr_save_dir, 'softmax')
print(lr_save_path)

max_vocab_size = 5000
seq_length = 600  # 输入x的维度
num_epochs = 100
batch_size = 32
print_per_batch = 700




if not os.path.exists(vocab_dir):
    build_vocab(train_dir, vocab_dir, max_vocab_size)
print('build vocab over')
# 全部分类，分类对应的id
categorys, cat_to_id = read_category()
print('read category over')
words, word_to_id = read_vocab(vocab_dir)
print('read vocab over')
x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, seq_length)
x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, seq_length)
print(len(x_train))
print(len(y_train))

print('process file over')
num_classes = len(cat_to_id)
# 定义模型

with tf.device('/cpu:0'):
    total_batch = 0  # 总批次
    best_acc_train = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练
    flag = False

    x = tf.placeholder(tf.float32, [None, seq_length], name='input_x')
    y_ = tf.placeholder(tf.float32, [None, num_classes], name='input_y')

    w = tf.Variable(tf.zeros([seq_length, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))
    # w = tf.Variable(tf.truncated_normal(shape=[seq_length, num_classes], mean=0, stddev=1))
    # b = tf.Variable(tf.truncated_normal(shape=[num_classes], mean=0, stddev=1))

    y_mat = tf.matmul(x, w) + b
    y = tf.nn.softmax(y_mat)
    #    cost = -tf.reduce_sum(y_*tf.log(y))  #交叉熵的计算方式
    #    cost = tf.reduce_sum(tf.square(y_-y))
    cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_mat)
    loss = tf.reduce_mean(cost)
    # cost = -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])
    # loss = tf.reduce_mean(cost)
    train_step1 = tf.train.AdamOptimizer()
    train_step = train_step1.minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # argmax是指取数组中最大的值所在的索引
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 配置Saver
    saver = tf.train.Saver()
    # 训练模型
    print("Training and evaluating...")
    print('initial')
    print('session')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            print('epoch:', epoch + 1)
            batch_train = batch_iter(x_train, y_train, batch_size)
#             # 这里x_batch的维度是(batch_size, seq_length), batch_size其实就是每次取的文档的个数
            for x_batch, y_batch in batch_train:
                # print(len(x_batch))
                # print(y_batch)
                if total_batch % print_per_batch == 0:
                    loss_train, acc_train = sess.run([loss, accuracy],
                                                     feed_dict={x: x_train, y_: y_train})
                    loss_val, acc_val = evaluate(sess, x_val, y_val)
                    if acc_train > best_acc_train:
                        print(best_acc_train)
                        print('保存模型')
                        # 保存最好结果
                        best_acc_train = acc_train
                        saver.save(sess=sess, save_path=lr_save_path)
                        improve_str = "*"
                    else:
                        improve_str = ""
                #            pdb.set_trace()
                    msg = 'iter:{},train_loss:{},train_acc:{},val_loss:{},val_acc:{}'
                    print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val))
                    print('-----------------------------------------------')
#
                sess.run(train_step, feed_dict={x: x_batch, y_: y_batch})
                if total_batch % print_per_batch == 0:
                    print('y', sess.run(tf.argmax(y, 1), feed_dict={x: x_batch, y_: y_batch}))
                    print('y_', sess.run(tf.argmax(y_, 1), feed_dict={x: x_batch, y_: y_batch}))
                total_batch += 1

