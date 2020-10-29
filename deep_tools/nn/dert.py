#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import logging.handlers
import os

import tensorflow as tf

from deep_tools.nn import Transformer

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
        self.query_pos_single_batch = tf.expand_dims(self.query_pos, 0)

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        # shape = (50, hidden_dim // 2)
        self.row_embed = tf.Variable(tf.random.uniform((50, hidden_dim // 2)))
        self.col_embed = tf.Variable(tf.random.uniform((50, hidden_dim // 2)))

        # for col_embed & row_embed concatenation
        self.concatenate = tf.keras.layers.Concatenate()

    @staticmethod
    def _box_cxcywh_to_xyxy(x):
        """
        Change predict boxes position representation
        :param x: (number of boxes, 4) predict boxes for one image
        :return: (number of boxes, 4)
        """
        x_c, y_c, w, h = tf.unstack(x, axis=1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return tf.stack(b, axis=1)

    def _rescale_bboxes(self, out_bbox, img_width, img_height):
        """
        Rescale predict boxes at the image size
        :param out_bbox: (number of boxes, 4) predict boxes for one image
        :param img_width: (int)
        :param img_height: (int)
        :return: (number of boxes, 4) x, y top left. x, y bottom rigth
        """
        b = self._box_cxcywh_to_xyxy(out_bbox)
        return b * tf.constant([img_width, img_height, img_width, img_height], dtype=tf.float32)

    def call(self, inputs, detection_ceil=0.7, training=False):
        """

        :param inputs: (batch size, image array, channel last)
            Note the size of all images need to be the same
        :param detection_ceil: (float) proba detection ceil
        :param training: (bool) used for dropout layers indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (doing nothing).
        :return: (dict) Dert prediction output and Transformer attention weights. Dict format
                {
                    'decoder_layer<X>block1': attention_weights softmax result
                        get from scaled_dot_product_attention of the block 1 decoder layer
                    'decoder_layer<X>block2': attention_weights softmax result
                        get from scaled_dot_product_attention of the block 2 decoder layer

                    "pred_logits": prediction labels probability shape is (batch size, number of boxes, number of classes + 1),
                    "pred_boxes": bouding box prediction shape is (batch size, number of boxes, 4),
                    "rescale_boxes": (list) shape (batch size, number of boxes, 4)
                        rescale boxes to the image size with confidence upper than detection_ceil (x, y top left. x, y bottom rigth)
                    "labels_confidence": (list) shape (batch size, number of boxes, number of classes)
                        Predict labels with confidence upper than detection_ceil
                }
        """

        # Extract image features with the pre train CNN model

        # Apply preprocess function
        if self.backbone_preprocess is not None:
            inputs = self.backbone_preprocess(inputs)

        # propagate inputs through backbone model and get output features
        x = self.backbone(inputs)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        batch, H, W, features_size = h.shape

        # shape: (batch, H*W, features). Need to be the same hase the query pos
        flatten_features = tf.reshape(h, (batch, H * W, self.hidden_dim))

        # Creat the positional encoding

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

        # Pass throw transformer and 2 dense layers

        # Creat the target the good quantity for the target sequence shape
        query_pos = tf.concat([tf.identity(self.query_pos_single_batch) for _ in range(batch)], axis=0)

        # Get attention sequence shape same has query pos (batch, 100, hidden_dim)
        decoder_out, attention_weights = self.transformer(flatten_pos + 0.1 * flatten_features, query_pos, training)

        pred_logits = self.linear_class(decoder_out)
        pred_boxes = self.linear_bbox(decoder_out)

        # Creat boxes

        # remove the last fake classe and keep only predictions with 0.7+ confidence for each images batch
        probas = pred_logits[:, :, :-1]
        bool_keep_boxes = tf.reduce_max(probas, -1) > detection_ceil

        rescale_boxes = []
        labels_confidence = []

        for img_index, bool_keep_box in enumerate(bool_keep_boxes):
            keep_box = tf.boolean_mask(pred_boxes[img_index], bool_keep_box)
            if len(keep_box) > 0:
                labels_confidence.append(tf.boolean_mask(probas[img_index], bool_keep_box))
                rescale_boxes.append(self._rescale_bboxes(keep_box, W, H))

        attention_weights.update({
            "pred_logits": probas,
            "pred_boxes": pred_boxes,
            "rescale_boxes": rescale_boxes,
            "labels_confidence": labels_confidence
        })
        return attention_weights
