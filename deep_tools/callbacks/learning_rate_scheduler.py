#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import logging.handlers
import os

import numpy as np
from tensorflow.keras.callbacks import LearningRateScheduler

PYTHON_LOGGER = logging.getLogger(__name__)
if not os.path.exists("log"):
    os.mkdir("log")
HDLR = logging.handlers.TimedRotatingFileHandler("log/learning_rate_scheduler.log",
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


class StepDecay(LearningRateScheduler):

    def __init__(self, init_learning_rate=0.01, factor=0.25, drop_every=5):
        self.init_learning_rate = init_learning_rate
        self.factor = factor
        self.drop_every = drop_every
        super().__init__(self.step_decay)

    def step_decay(self, epoch):
        # compute learning rate for the current epoch
        alpha = self.init_learning_rate * (self.factor ** np.floor((1 + epoch) / self.drop_every))
        # return the learning rate
        return float(alpha)
