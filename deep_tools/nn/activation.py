#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
import os
import logging.handlers
import tensorflow as tf

from deep_tools.nn.functional import scaled_dot_product_attention

PYTHON_LOGGER = logging.getLogger(__name__)
if not os.path.exists("log"):
    os.mkdir("log")
HDLR = logging.handlers.TimedRotatingFileHandler("log/activation.log",
                                                 when="midnight", backupCount=60)
STREAM_HDLR = logging.StreamHandler()
FORMATTER = logging.Formatter("%(asctime)s %(filename)s [%(levelname)s] %(message)s")
HDLR.setFormatter(FORMATTER)
STREAM_HDLR.setFormatter(FORMATTER)
PYTHON_LOGGER.addHandler(HDLR)
PYTHON_LOGGER.addHandler(STREAM_HDLR)
PYTHON_LOGGER.setLevel(logging.DEBUG)

# Absolute path to the folder location of this python file
FOLDER_ABSOLUTE_PATH = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention consists of four parts:
        - Linear layers and split into heads.
        - Scaled dot-product attention.
        - Concatenation of heads.
        - Final linear layer.
    Each multi-head attention block gets three inputs; Q (query), K (key), V (value).
    These are put through linear (Dense) layers and split up into multiple heads.

    The scaled_dot_product_attention defined above is applied to each head (broadcasted for efficiency).
    An appropriate mask must be used in the attention step.
    The attention output for each head is then concatenated and put through a final Dense layer.

    Instead of one single attention head, Q, K, and V are split into multiple heads because it allows the model
    to jointly attend to information at different positions from different representational spaces.
    After the split each head has a reduced dimensionality,
    so the total computation cost is the same as a single head attention with full dimensionality.

    >>> temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    >>> y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    >>> out, attn = temp_mha(y, k=y, q=y, mask=None)
    >>> out.shape, attn.shape
    """

    def __init__(self, d_model, num_heads):
        """
        :param d_model: (int) the number of expected features in the input (required).
        :param num_heads: (int) the number of heads in the multiheadattention models (required).
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights
