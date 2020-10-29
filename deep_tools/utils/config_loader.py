#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import configparser
import logging.handlers
import os
from typing import Text

from deep_tools.utils.files_tools import read_json_file

PYTHON_LOGGER = logging.getLogger(__name__)
if not os.path.exists("log"):
    os.mkdir("log")
HDLR = logging.handlers.TimedRotatingFileHandler("log/config_loader.log",
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


class JsonConfigLoader:

    def __init__(self, config_path: Text):
        self.config_path = config_path
        self.config = read_json_file(config_path)
        try:
            self.__dict__.update(self.config)
        except Exception as e:
            PYTHON_LOGGER.error("Error to load the configurations: {}".format(e))

    def get(self, name, default=None):
        value = self.__dict__.get(name, default)
        if value is None:
            raise KeyError("The name {} not in the config file {}".format(name, self.config_path))
        return value
