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
        self.linear_class = tf.keras.layers.Dense(num_classes + 1, input_shape=(hidden_dim,))
        self.linear_bbox = tf.keras.layers.Dense(4, input_shape=(hidden_dim,))

        # output positional encodings (object queries)
        self.query_pos = tf.Variable(tf.random.uniform((100, hidden_dim)))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        # shape = (50, hidden_dim // 2)
        self.row_embed = tf.Variable(tf.random.uniform((50, hidden_dim // 2)))
        self.col_embed = tf.Variable(tf.random.uniform((50, hidden_dim // 2)))

        # for col_embed & row_embed concatenation
        self.concatenate = tf.keras.layers.Concatenate()

    def call(self, inputs):
        # Apply preprocess function
        if self.backbone_preprocess is not None:
            inputs = self.backbone_preprocess(inputs)

        print("b call")
        # propagate inputs through backbone model and get output features
        x = self.backbone(inputs)
        print("e call")

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
        # TODO flatten
        """
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),start
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)

        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h),
                'pred_boxes': self.linear_bbox(h).sigmoid()}
        """
