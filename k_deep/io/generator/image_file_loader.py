#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import logging.handlers
import os

import cv2

from k_deep.io.generator.generator import Generator

PYTHON_LOGGER = logging.getLogger(__name__)
if not os.path.exists("log"):
    os.mkdir("log")
HDLR = logging.handlers.TimedRotatingFileHandler("log/image_file_loader.log",
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


class ImageFileLoader(Generator):

    def __init__(self, image_paths, preprocessors=None):
        """

        :param image_paths:
        :param preprocessors: (list of preprocessing class)
        """
        # store the image preprocessor
        self.preprocessors = preprocessors

        # if the preprocessors are None, initialize them as an
        # empty list
        if self.preprocessors is None:
            self.preprocessors = []

        self.image_paths = image_paths
        self._i = 0

    def _next(self):
        image_path = self.image_paths[self._i]
        self._i += 1
        image = cv2.imread(image_path)
        # check to see if our preprocessors are not None
        if self.preprocessors is not None:
            # loop over the preprocessors and apply each to
            # the image
            for p in self.preprocessors:
                image = p.preprocess(image)
        return image

    def _stop_iteration(self):
        return self._i == len(self.image_paths)
