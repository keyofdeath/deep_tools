#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
import os
import logging.handlers

import tensorflow as tf

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
                 backbone=tf.keras.applications.ResNet50(weights="imagenet", include_top=False),
                 backbone_preprocess=tf.keras.applications.imagenet_utils.preprocess_input,
                 hidden_dim=256):
        super().__init__(Dert, self)
        self.backbone = backbone
        self.backbone_preprocess = backbone_preprocess

        self.conv = tf.keras.layers.Conv2D(hidden_dim, 1, padding="same",
                                           input_shape=(None, None, self.backbone.output_shape[-1]))

    def call(self, x, **kwargs):
        pass
