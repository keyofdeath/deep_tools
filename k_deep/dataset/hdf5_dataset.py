#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import logging.handlers
import os

import tensorflow as tf

PYTHON_LOGGER = logging.getLogger(__name__)
if not os.path.exists("log"):
    os.mkdir("log")
HDLR = logging.handlers.TimedRotatingFileHandler("log/tf_hdf5_dataset.log",
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


class TfHdf5Dataset(tf.data.Dataset):

    def _shape_invariant_to_type_spec(self, shape):
        pass

    def _inputs(self):
        pass

    def element_spec(self):
        pass

    def _generator(self, preprocessors, ):
        # Opening the file
        time.sleep(0.03)

        for sample_idx in range(num_samples):
            # Reading data (line, record) from the file
            time.sleep(0.015)

            yield (sample_idx,)

    def __new__(self, file_path, preprocessors, limit_size=None):

        self.file_path = file_path
        self.preprocessors = preprocessors
        self.limit_size = limit_size
        return tf.data.Dataset.from_generator(
            self._generator,
            output_types=tf.dtypes.int64,
            output_shapes=(1,),
            args=(preprocessors, limit_size)
        )

class Hdf5DatasetCreator:

    def __init__(self, ):