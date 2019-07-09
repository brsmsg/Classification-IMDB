# -*- coding: utf-8 -*-
from model import *
from dataHelper import *
from sklearn import metrics
import os
import numpy as np
import time
from datetime import timedelta

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

save_path = 'D:\\DataSet\\IMDB\\aclImdb\\model'


def batch_iter(x, y, pos, batch_size):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    # pos = [i for i in range(model.config.seq_length)]

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    pos_shuffle = pos[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id], pos_shuffle[start_id:end_id]


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, pos_batch, keep_prob):

    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.input_pos: pos_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_, pos):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, pos, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch, pos_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, pos_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    # if not os.path.exists(save_dir):
    # os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    # x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    # x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    #x_train, y_train, pos_train = prepare_senti_data(POS_PATH, NEG_PATH, word_to_id, cat_to_id, config.seq_length, 5000)
    x_train, y_train, pos_train = prepare_data(POS_PATH, NEG_PATH, word_to_id, cat_to_id, config.seq_length, 5000)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0              # 总批次
    best_acc_val = 0.0           # 最佳验证集准确率
    last_improved = 0            # 记录上一次提升批次
    require_improvement = 1000   # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, pos_train, config.batch_size)
        for x_batch, y_batch, pos_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch,pos_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                # feed_dict[model.keep_prob] = 1.0

                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                # loss_val, acc_val = evaluate(session, x_val, y_val)   # todo

                if acc_train > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_train
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)

                time_dif = get_time_dif(start_time)
                print(total_batch, time_dif, loss_train, acc_train)
                # msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},'\
                #    + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                # print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环

        if flag:  # 同上
            break
        # saver.save(sess=session, save_path=save_path)


def test():
    # start_time = time.time()
    #x_test, y_test, pos_test = prepare_senti_data(TEST_POS_PATH, TEST_NEG_PATH, word_to_id, cat_to_id, config.seq_length, 1000)
    x_test, y_test, pos_test = prepare_data(TEST_POS_PATH, TEST_NEG_PATH, word_to_id, cat_to_id, config.seq_length, 1000)


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess = sess, save_path=save_path) # 读取保存的模型
    print('testing..')

    loss_test, acc_test = evaluate(sess, x_test, y_test, pos_test)
    print(loss_test, acc_test)

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1)/batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)    # 预测结果

    for i in range(num_batch):
        start_id = i*batch_size
        end_id = min((i+1)*batch_size, data_len)
        feed_dict = {
            model.input_x:x_test[start_id:end_id],
            model.input_pos:pos_test[start_id:end_id],
            model.keep_prob:1.0
        }
        y_pred_cls[start_id:end_id] = sess.run(model.y_pred_cls, feed_dict=feed_dict)
    # print(y_pred_cls.shape, y_test_cls.shape)

    y_pred_cls = [id_to_cat(id) for id in y_pred_cls]
    y_test_cls = [id_to_cat(id) for id in y_test_cls]
    print(y_pred_cls)
    print(y_test_cls)
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))


if __name__ == '__main__':
    config = TCNNConfig()
    categories, cat_to_id = prepare_cat()
    #words, word_to_id = prepare_vocab(VOCAB_PATH)
    words, word_to_id = prepare_vocab(VOCAB_PATH)
    config.vocab_size = len(words)
    model = TextCNN(config)

    train()
    test()