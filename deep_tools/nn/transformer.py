#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import logging.handlers
import os

import tensorflow as tf

from deep_tools.nn.activation import MultiHeadAttention

PYTHON_LOGGER = logging.getLogger(__name__)
if not os.path.exists("log"):
    os.mkdir("log")
HDLR = logging.handlers.TimedRotatingFileHandler("log/transformer.log",
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


class Transformer(tf.keras.layers.Layer):
    """
    A transformer model.
    The architecture is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer,Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
    model with corresponding parameters.
    """

    def __init__(self, num_encoder_layers, num_decoder_layers,
                 d_model, num_heads, dim_feedforward=2048, dropout=0.1, activation="relu",
                 custom_encoder=None, custom_decoder=None):
        """
        :param num_encoder_layers: (int) Number of TransformerEncoderLayer (required)
        :param num_decoder_layers: (int) Number of TransformerDecoderLayer (required)
        :param d_model: (int) the number of expected features in the input (required).
        :param num_heads: (int) the number of heads in the multiheadattention models (required).
        :param dim_feedforward: (int) the dimension of the feedforward network model (default=2048).
        :param dropout: (double) the dropout value (default=0.1).
        :param activation: (string) the activation function of intermediate layer, relu or gelu (default=relu).
        :param custom_encoder: (tf.keras.layers.Layer) custom encoder (default=None).
        :param custom_decoder: (tf.keras.layers.Layer) custom decoder (default=None).
        """
        super(Transformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, dim_feedforward, dropout,
                                              activation, encoder_norm)
        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, dim_feedforward, dropout,
                                              activation, decoder_norm)

    def call(self, inp, tar, training, enc_padding_mask=None, look_ahead_mask=None, dec_padding_mask=None):
        """

        :param inp: (Tensor) the sequence to the encoder layer (required).
            Shape should be: (batch size, source sequence length, feature number)

        :param tar: (Tensor) the target sequence send to the decoder layer (required).
            Shape should be: (batch size, target sequence length, feature number)

        :param training: (bool) used for dropout layers indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (doing nothing).

        :param enc_padding_mask: the mask for the inp sequence (optional).
        :param look_ahead_mask: the mask for the tar sequence (optional).
        :param dec_padding_mask: the mask for the encoder_output sequence second MultiHeadAttention (optional).
        :return: (tuple) decoder output, Attention weights
            Attention weights: (dict) format
                {
                    'decoder_layer<X>block1': attention_weights softmax result
                        get from scaled_dot_product_attention of the block 1 decoder layer
                    'decoder_layer<X>block2': attention_weights softmax result
                        get from scaled_dot_product_attention of the block 2 decoder layer
                }
            Where X is the decoder layer number
        """
        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, training, enc_padding_mask)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        return dec_output, attention_weights


class TransformerEncoder(tf.keras.layers.Layer):
    """
    TransformerEncoder is a stack of N encoder layers

    >>> sample_encoder = TransformerEncoder(4, 512, 8, 2048)
    >>> x = tf.random.uniform((64, 43, 512))
    >>> sample_encoder_output = sample_encoder(x, False, None)
    >>> sample_encoder_output.shape  # (batch_size, input_seq_len, d_model)
    """

    def __init__(self, num_layers, d_model, num_heads, dim_feedforward=2048, dropout=0.1, activation="relu", norm=None):
        """
        :param num_layers: (int) Number of TransformerEncoderLayer (required)
        :param d_model: (int) the number of expected features in the input (required).
        :param num_heads: (int) the number of heads in the multiheadattention models (required).
        :param dim_feedforward: (int) the dimension of the feedforward network model (default=2048).
        :param dropout: (double) the dropout value (default=0.1).
        :param activation: (string) the activation function of intermediate layer, relu or gelu (default=relu).
        :param norm: (layer) the layer normalization component (optional).
        """
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.enc_layers = [TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout, activation)
                           for _ in range(num_layers)]

        self.norm = norm

    def call(self, x, training, mask=None):
        """
        :param x: (Tensor) the sequence to the encoder layer (required).
        :param training: (bool) used for dropout layers indicating whether the layer should behave in
                    training mode (adding dropout) or in inference mode (doing nothing).
        :param mask: the mask for the x sequence (optional).
        :return:

        Shape:
            see the docs in Transformer class.
        """
        out = x
        for i in range(self.num_layers):
            out = self.enc_layers[i](out, training, mask)

        if self.norm is not None:
            out = self.norm(out)

        # (batch_size, input_seq_len, d_model)
        return out


class TransformerDecoder(tf.keras.layers.Layer):
    """
    TransformerDecoder is a stack of N decoder layers

    Encoder
    >>> sample_encoder = TransformerEncoder(4, 512, 8, 2048)
    >>> x = tf.random.uniform((64, 43, 512))
    >>> sample_encoder_output = sample_encoder(x, False, None)
    >>> sample_encoder_output.shape

    Decoder
    >>> sample_decoder = TransformerDecoder(512, 8, 2048)
    >>> x = tf.random.uniform((64, 43, 512)) # shape 64, 43, 512
    >>> sample_decoder_output, attn  = sample_decoder(x, sample_encoder_output, False, None, None)
    >>> sample_decoder_output.shape, attn['decoder_layer2_block2'].shape
    """

    def __init__(self, num_layers, d_model, num_heads, dim_feedforward=2048, dropout=0.1, activation="relu", norm=None):
        """
        :param num_layers: (int) Number of TransformerDecoderLayer (required)
        :param d_model: (int) the number of expected features in the input (required).
        :param num_heads: (int) the number of heads in the multiheadattention models (required).
        :param dim_feedforward: (int) the dimension of the feedforward network model (default=2048).
        :param dropout: (double) the dropout value (default=0.1).
        :param activation: (string) the activation function of intermediate layer, relu or gelu (default=relu).
        :param norm: (layer) the layer normalization component (optional).
        """
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.dec_layers = [TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout, activation)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout)

        self.norm = norm

    def call(self, x, encoder_output, training, look_ahead_mask=None, padding_mask=None):
        """
        :param x: (Tensor) Target sequence, the sequence to the decoder layer (required).
        :param encoder_output: (Tensor) the sequence from the last layer of the encoder (required).
        :param training: (bool) used for dropout layers indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (doing nothing).
        :param look_ahead_mask: the mask for the x sequence (optional).
        :param padding_mask: the mask for the encoder_output sequence second MultiHeadAttention (optional).
        :return: (tuple) decoder output, Attention weights
            Attention weights: (dict) format
                {
                    'decoder_layer<X>block1': attention_weights softmax result
                        get from scaled_dot_product_attention of the block 1 decoder layer
                    'decoder_layer<X>block2': attention_weights softmax result
                        get from scaled_dot_product_attention of the block 2 decoder layer
                }
            Where X is the decoder layer number

        Shape:
            see the docs in Transformer class.
        """
        out = x
        attention_weights = {}
        for i in range(self.num_layers):
            out, block1, block2 = self.dec_layers[i](out, encoder_output, training,
                                                     look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        if self.norm is not None:
            out = self.norm(out)

        # x.shape == (batch_size, target_seq_len, d_model)
        return out, attention_weights


class TransformerEncoderLayer(tf.keras.layers.Layer):
    """
    Each encoder layer consists of sublayers:
        1. Multi-head attention (with padding mask)
        2. Point wise feed forward networks.

    Each of these sublayers has a residual connection around it followed by a layer normalization.
    Residual connections help in avoiding the vanishing gradient problem in deep networks.

    The output of each sublayer is LayerNorm(x + Sublayer(x)).
    The normalization is done on the d_model (last) axis. There are N encoder layers in the transformer.

    >>> sample_encoder_layer = TransformerEncoderLayer(512, 8, 2048)
    >>> x = tf.random.uniform((64, 43, 512))
    >>> sample_encoder_layer_output = sample_encoder_layer(x, False, None)
    >>> sample_encoder_layer_output.shape  # (batch_size, input_seq_len, d_model)
    """

    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1, activation="relu"):
        """
        :param d_model: (int) the number of expected features in the input (required).
        :param num_heads: (int) the number of heads in the multiheadattention models (required).
        :param dim_feedforward: (int) the dimension of the feedforward network model (default=2048).
        :param dropout: (double) the dropout value (default=0.1).
        :param activation: (string) the activation function of intermediate layer, relu or gelu (default=relu).
        """
        super(TransformerEncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dim_feedforward, dropout, activation)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training, mask=None):
        """
        :param x: (Tensor) the sequence to the encoder layer (required).
        :param training: (bool) used for dropout layers indicating whether the layer should behave in
                    training mode (adding dropout) or in inference mode (doing nothing).
        :param mask: the mask for the x sequence (optional).
        :return:

        Shape:
            see the docs in Transformer class.
        """
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class TransformerDecoderLayer(tf.keras.layers.Layer):
    """
    Each decoder layer consists of sublayers:
        1. Masked multi-head attention (with look ahead mask and padding mask)
        2. Multi-head attention (with padding mask).
            V (value) and K (key) receive the encoder output as inputs.
            Q (query) receives the output from the masked multi-head attention sublayer.
        3. Point wise feed forward networks

        Each of these sublayers has a residual connection around it followed by a layer normalization.
        The output of each sublayer is LayerNorm(x + Sublayer(x)).
        The normalization is done on the d_model (last) axis.

        There are N decoder layers in the transformer.

    Encoder part
    >>> sample_encoder_layer = TransformerEncoderLayer(512, 8, 2048)
    >>> x = tf.random.uniform((64, 43, 512))
    >>> sample_encoder_layer_output = sample_encoder_layer(x, False, None) # (batch_size, input_seq_len, d_model)

    Decoder part
    >>> sample_encoder_layer = TransformerDecoderLayer(512, 8, 2048)
    >>> x = tf.random.uniform((64, 43, 512)) # shape 64, 43, 512
    >>> sample_decoder_layer_output, _, _ = sample_encoder_layer(x, sample_encoder_layer_output, False, None, None)
    >>> sample_decoder_layer_output.shape # (batch_size, target_seq_len, d_model)
    """

    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1, activation="relu"):
        """
        :param d_model: (int) the number of expected features in the input (required).
        :param num_heads: (int) the number of heads in the multiheadattention models (required).
        :param dim_feedforward: (int) the dimension of the feedforward network model (default=2048).
        :param dropout: (double) the dropout value (default=0.1).
        :param activation: (string) the activation function of intermediate layer, relu or gelu (default=relu).
        """
        super(TransformerDecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dim_feedforward, dropout, activation)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)

    def call(self, x, encoder_output, training, look_ahead_mask=None, padding_mask=None):
        """
        :param x: (Tensor) Target sequence shape should be (batch_size, seq length, d_model) (required).
        :param encoder_output: (Tensor) the sequence from the last layer of the encoder (required).
        :param training: (bool) used for dropout layers indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (doing nothing).
        :param look_ahead_mask: the mask for the x sequence (optional).
        :param padding_mask: the mask for the encoder_output sequence second MultiHeadAttention (optional).
        :return: (tuple) decoder layer output, Attention weights block 1, Attention weights block 2

        Shape:
            see the docs in Transformer class.
        """
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            encoder_output, encoder_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


def point_wise_feed_forward_network(d_model, dim_feedforward, dropout=0.1, activation="relu"):
    """
    Point wise feed forward network consists of two fully-connected layers
    with a activation function and dropout in between.
    >>> sample_ffn = point_wise_feed_forward_network(512, 2048)
    >>> sample_ffn(tf.random.uniform((64, 50, 512))).shape
    :param d_model: (int) the number of expected features in the input
    :param dim_feedforward: (int) the dimension of the feedforward network model.
    :param dropout: (int) the dropout value (default=0.1).
    :param activation: (string) the activation function of intermediate layer, relu or gelu (default=relu).
    :return: (Model) Dense -> activation -> Dropout -> Dense
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dim_feedforward, activation=activation),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])
