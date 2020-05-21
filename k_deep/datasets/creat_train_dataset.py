#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import logging.handlers
import os

from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from tensorflow.keras.utils import to_categorical

from k_deep.datasets import DatasetLoader
from k_deep.preprocessing import ResizePreprocessor, ImageToArrayPreprocessor

PYTHON_LOGGER = logging.getLogger(__name__)
if not os.path.exists("log"):
    os.mkdir("log")
HDLR = logging.handlers.TimedRotatingFileHandler("log/creat_train_dataset.log",
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


class ImageCreatTrainDataset:
    """
    Load all images and process all images for model training
        load the image and extract the class label assuming
        that our path has the following format:
        /path/to/dataset/{class}/{image}.jpg
    """

    def __init__(self, dataset_path, img_dim, test_size):
        self.dataset_path = dataset_path
        self.img_dim = img_dim
        self.test_size = test_size
        # Get all images list
        self.image_paths = list(paths.list_images(self.dataset_path))

        # initialize the image preprocessors
        rp = ResizePreprocessor(self.img_dim, self.img_dim)
        iap = ImageToArrayPreprocessor()
        self.dataset_loader = DatasetLoader(preprocessors=[rp, iap])
        self.dataset = None
        self.labels = None
        self.nb_labels = None
        self.label_list = None
        self.train_x, self.train_y = None, None
        self.test_x, self.test_y = None, None

    def load_dataset(self):
        """

        :return:
        """
        (self.dataset, self.labels) = self.dataset_loader.load(self.image_paths, verbose=500)
        norm_dataset = self.dataset.astype("float") / 255.0
        self.nb_labels = len(set(self.labels))

        # partition the dataset into training and testing splits using 75% of
        # the dataset for training and the remaining 25% for testing
        (self.train_x, self.test_x, self.train_y, self.test_y) = train_test_split(norm_dataset, self.labels,
                                                                                  test_size=self.test_size,
                                                                                  random_state=42)
        encoder = LabelEncoder()
        self.train_y = encoder.fit_transform(self.train_y)
        self.test_y = encoder.transform(self.test_y)
        # convert the labels from integers to vectors
        self.train_y = to_categorical(self.train_y, self.nb_labels)
        self.test_y = to_categorical(self.test_y, self.nb_labels)
        self.label_list = list(encoder.classes_)

    def get_train_data(self):
        """

        :return: (tuple of np array) train images, train labels
        """
        assert self.train_x is not None and self.train_y is not None
        return self.train_x, self.train_y

    def get_test_data(self):
        """

        :return: (tuple of np array) test images, test labels
        """
        assert self.test_x is not None and self.test_y is not None
        return self.test_x, self.test_y

    def get_labels(self):
        return self.label_list, self.nb_labels
