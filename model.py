# -*- coding: utf-8 -*-
import tensorflow as tf


class TCNNConfig(object):
    """CNN配置参数"""
    embedding_dim = 256  # 词向量维度
    seq_length = 128 # 序列长度
    num_classes = 2  # 类别数
    num_filters = 100  # 卷积核数目
    pos_embedding_dim = 100  # position_embedding维度

    # kernel_size = 5         # 卷积核尺寸
    kernel_sizes = [2, 3, 4]
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率
    regular_rate = 1e-4  # l2正则

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.input_pos = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_pos')
        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs1 = tf.nn.embedding_lookup(embedding, self.input_x)  # [None, seq_length, embedding_dim]

            pos_embedding = tf.get_variable('pos_embedding', [self.config.vocab_size, 5])
            embedding_inputs2 = tf.nn.embedding_lookup(pos_embedding, self.input_pos)

            embedding_inputs = tf.concat([embedding_inputs1, embedding_inputs2], -1)

        with tf.name_scope("cnn"):
            """
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
            """
            gmp = []
            for i, size in enumerate(self.config.kernel_sizes):
                conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, size)
                pooled = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
                gmp.append(pooled)
            gmp = tf.concat(gmp, 1)

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            self.loss_reg = self.loss + tf.contrib.layers.apply_regularization(
                tf.contrib.layers.l2_regularizer(self.config.regular_rate),
                [v for v in tf.trainable_variables() if 'bias' not in v.name])

            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss_reg)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
