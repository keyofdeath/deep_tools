#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import logging.handlers
import os

import tensorflow as tf

from k_deep.nn import Transformer

PYTHON_LOGGER = logging.getLogger(__name__)
if not os.path.exists("log"):
    os.mkdir("log")
HDLR = logging.handlers.TimedRotatingFileHandler("log/dert.log",
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


class Dert(tf.keras.layers.Layer):

    def __init__(self, num_classes,
                 transformer: Transformer,
                 backbone=tf.keras.applications.ResNet50(weights="imagenet", include_top=False),
                 backbone_preprocess=tf.keras.applications.imagenet_utils.preprocess_input,
                 hidden_dim=256):
        super(Dert, self).__init__()

        self.hidden_dim = hidden_dim

        self.backbone = backbone
        self.backbone_preprocess = backbone_preprocess

        self.transformer = transformer

        # create conversion layer
        # now the model will take as input arrays of shape (*, self.backbone.output_shape[-1])
        self.conv = tf.keras.layers.Conv2D(hidden_dim, 1, padding="same",
                                           input_shape=(self.backbone.output_shape[-1],))

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        # now the model will take as input arrays of shape (*, hidden_dim)
        self.linear_class = tf.keras.layers.Dense(num_classes + 1, input_shape=(hidden_dim,), activation="softmax")
        self.linear_bbox = tf.keras.layers.Dense(4, input_shape=(hidden_dim,), activation="sigmoid")

        # output positional encodings (object queries)
        self.query_pos = tf.Variable(tf.random.uniform((100, hidden_dim)))
        # Shape 1, 100, hidden_dim
        self.query_pos = tf.expand_dims(self.query_pos, 0)

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        # shape = (50, hidden_dim // 2)
        self.row_embed = tf.Variable(tf.random.uniform((50, hidden_dim // 2)))
        self.col_embed = tf.Variable(tf.random.uniform((50, hidden_dim // 2)))

        # for col_embed & row_embed concatenation
        self.concatenate = tf.keras.layers.Concatenate()

    def call(self, inputs, detection_ceil=0.7, training=False):
        """

        :param inputs: (batch size, image array, channel last)
        :param detection_ceil: (float) proba detection ceil
        :param training:
        :return: (dict) Dert prediction output and Transformer attention weights. Dict format
                {
                    'decoder_layer<X>block1': attention_weights softmax result
                        get from scaled_dot_product_attention of the block 1 decoder layer
                    'decoder_layer<X>block2': attention_weights softmax result
                        get from scaled_dot_product_attention of the block 2 decoder layer

                    "pred_logits": prediction labels shape is (batch size, seq length, number of classes + 1),
                    "pred_boxes": bouding box prediction shape is (batch size, seq length, 4)
                }
        """
        # Apply preprocess function
        if self.backbone_preprocess is not None:
            inputs = self.backbone_preprocess(inputs)

        # propagate inputs through backbone model and get output features
        x = self.backbone(inputs)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        batch, H, W, features_size = h.shape

        # Extract sub embed size of W, H to creat an array shape of (H, W, hidden_dim // 2)
        # shape = 1, W, hidden_dim // 2
        expand_col = tf.expand_dims(self.col_embed[:W], 0)
        # shape = H, 1, hidden_dim // 2
        expand_row = tf.expand_dims(self.row_embed[:H], 1)

        # Now we creat 2 array of shape (H, W, hidden_dim // 2)
        tile_expand_col = tf.tile(expand_col, (H, 1, 1))
        tile_expand_row = tf.tile(expand_row, (1, W, 1))

        # Concat this to array to have (H, W, hidden_dim) array
        pos = tf.concat([tile_expand_col, tile_expand_row], -1)
        flatten_pos = tf.reshape(pos, (H * W, self.hidden_dim))
        # Set position to shape 1, H*W, hidden_dim
        flatten_pos = tf.expand_dims(flatten_pos, 0)

        # shape: (batch, H*W, features). Need to be the same hase the query pos
        flatten_features = tf.reshape(h, (batch, H * W, self.hidden_dim))

        # Get attention sequence shape same has query pos (batch, 100, hidden_dim)
        h, attention_weights = self.transformer(flatten_pos + 0.1 * flatten_features, self.query_pos, training)

        pred_logits = self.linear_class(h)
        pred_boxes = self.linear_bbox(h)

        # remove the last fake classe and keep only predictions with 0.7+ confidence
        probas = pred_logits[:, :, :-1]

        keep = tf.reduce_max(probas, -1) > detection_ceil
        for img_index, proba_keep in enumerate(keep):
            keep_bbox = tf.boolean_mask(pred_boxes[img_index], proba_keep)
            # TODO rescale_bboxes
        attention_weights.update({
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,

        })

        return attention_weights
