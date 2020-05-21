#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import logging.handlers
import os
from typing import Text

import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint

from k_deep.callbacks import TrainingMonitor
from k_deep.callbacks.learning_rate_scheduler import StepDecay
from k_deep.datasets.creat_train_dataset import ImageCreatTrainDataset
from k_deep.nn.conv import ShallowNet, LeNet, MiniVGGNet
from k_deep.preprocessing import ImageToArrayPreprocessor
from k_deep.preprocessing.resizepreprocessor import ResizePreprocessor
from k_deep.utils import JsonConfigLoader
from k_deep.utils.evaluation import training_evaluation, classification_show
from k_deep.utils.files_tools import read_json_file, save_json_file, creat_folder_if_not_exist

PYTHON_LOGGER = logging.getLogger(__name__)
if not os.path.exists("log"):
    os.mkdir("log")
HDLR = logging.handlers.TimedRotatingFileHandler("log/image_classification.log",
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


class ImageClassificationTrain:
    MODEL_MAPPING = {
        ShallowNet.name(): ShallowNet,
        LeNet.name(): LeNet,
        MiniVGGNet.name(): MiniVGGNet
    }

    def __init__(self, config_path: Text):
        """

        :param config_path:
        """
        self.config = JsonConfigLoader(config_path)
        self.img_dim = self.config.get("img_dim", 32)
        self.test_size = self.config.get("test_size", 25) / 100
        self.batch_size = self.config.get("batch_size", 32)
        self.epochs = self.config.get("epochs", 100)
        self.learning_rate = self.config.get("learning_rate", 0.005)
        self.verbose = self.config.get("verbose", 1)

        self.model_name = self.config.get("model_name")
        self.dataset_path = self.config.get("dataset_path")

        self.monitoring = self.config.get("monitoring", True)
        self.monitoring_folder = self.config.get("monitoring_folder", "monitoring")

        self.checkpoint_saving = self.config.get("checkpoint_saving", True)
        self.weight_folder = self.config.get("weights_folder", os.path.join("weights", "best"))

        self.__model_list = ', '.join(list(self.MODEL_MAPPING.keys())).strip()
        if self.model_name not in self.MODEL_MAPPING:
            raise ValueError("The model name {} not in the list: {}".format(self.model_name, self.__model_list))

        self.save_model_name = self.config.get("save_model_name", "{}_ep{}_bs{}".format(self.model_name,
                                                                                        self.epochs,
                                                                                        self.batch_size))
        self.model_class = self.MODEL_MAPPING[self.model_name]

        self.dataset_image = ImageCreatTrainDataset(self.dataset_path, self.img_dim, self.test_size)
        self.dataset_image.load_dataset()

        self.train_x, self.train_y = self.dataset_image.get_train_data()
        self.test_x, self.test_y = self.dataset_image.get_test_data()
        self.label_list, self.nb_labels = self.dataset_image.get_labels()

        opt = SGD(lr=self.learning_rate, momentum=0.9, nesterov=True)
        self.model = self.model_class.build(width=self.img_dim, height=self.img_dim,
                                            depth=3, classes=self.nb_labels)
        loss = "categorical_crossentropy" if self.nb_labels > 2 else "binary_crossentropy"
        self.model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])

        # construct the set of callbacks
        self.callbacks = [StepDecay(init_learning_rate=self.learning_rate)]

        if self.monitoring:
            # Creat train monitoring callback
            creat_folder_if_not_exist(self.monitoring_folder)
            figPath = os.path.sep.join([self.monitoring_folder, "{}.png".format(os.getpid())])
            jsonPath = os.path.sep.join([self.monitoring_folder, "{}.json".format(os.getpid())])
            self.callbacks.append(TrainingMonitor(figPath, jsonPath=jsonPath))

        if self.checkpoint_saving:
            creat_folder_if_not_exist(self.weight_folder)
            # construct the callback to save only the *best* model to disk
            # based on the validation loss
            best_weights = os.path.join(self.weight_folder, "{}.hdf5".format(self.save_model_name))
            checkpoint = ModelCheckpoint(best_weights, monitor="val_loss", save_best_only=True, verbose=1)
            self.callbacks.append(checkpoint)

    def train_model(self, display_evaluation=False, nb_test_images=10):
        """

        :param display_evaluation:
        :param nb_test_images:
        :return:
        """
        PYTHON_LOGGER.info("Program pid {}".format(os.getpid()))
        PYTHON_LOGGER.info("Start training the model: {}, with {} epochs, {} batch_size".format(self.model_name,
                                                                                                self.epochs,
                                                                                                self.batch_size))
        nb_test_images = min(nb_test_images, len(self.test_x))
        H = self.model.fit(self.train_x, self.train_y,
                           validation_data=(self.test_x, self.test_y),
                           batch_size=self.batch_size,
                           epochs=self.epochs,
                           verbose=self.verbose,
                           callbacks=self.callbacks)

        # Save the model
        self.model.save("{}.h5".format(self.save_model_name))
        save_json_file({"img_dim": self.img_dim,
                        "label_list": self.label_list}, "{}.json".format(self.save_model_name))
        if not display_evaluation:
            return

        # evaluate the network
        PYTHON_LOGGER.info("Evaluating network")
        predictions = self.model.predict(self.test_x, batch_size=self.batch_size)
        print(classification_report(self.test_y.argmax(axis=1),
                                    predictions.argmax(axis=1),
                                    target_names=self.label_list))

        training_evaluation(H)
        norm_test_images = self.test_x[:nb_test_images]
        predictions = self.model.predict(norm_test_images).argmax(axis=1)
        test_images = [(img * 255).astype("uint8") for img in norm_test_images]
        classification_show(test_images, predictions, self.label_list, save_plot="test_{}.png".format(self.save_model_name))


class ImageClassificationLoad:

    def __init__(self, model_name, load_model_weight=None):
        """

        :param model_name:
        """
        if load_model_weight is None:
            self.model = load_model("{}.h5".format(model_name))
        else:
            self.model = load_model(load_model_weight)
        try:
            self.label_list = read_json_file("{}.json".format(model_name))
            self.img_dim = self.label_list["img_dim"]
            self.label_list = self.label_list["label_list"]
        except Exception:
            PYTHON_LOGGER.error("Error to load label list")
            self.label_list = None
            self.img_dim = self.model.input.shape[1]
            self.label_list = None
        # initialize the image preprocessors
        self.rp = ResizePreprocessor(self.img_dim, self.img_dim)
        self.iap = ImageToArrayPreprocessor()

    def predict(self, image):
        """

        :param image:
        :return:
        """
        norm_image = self.rp.preprocess(image)
        norm_image = self.iap.preprocess(norm_image)
        norm_image = np.array([norm_image.astype("float") / 255.0])
        prediction_index = self.model.predict([norm_image])
        return self.label_list[prediction_index.argmax(axis=1)[0]] if self.label_list is not None else prediction_index[0]
