#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import logging.handlers
import os

import matplotlib.pyplot as plt
import numpy as np
import cv2

PYTHON_LOGGER = logging.getLogger(__name__)
if not os.path.exists("log"):
    os.mkdir("log")
HDLR = logging.handlers.TimedRotatingFileHandler("log/evaluation.log",
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


def training_evaluation(H, save_plot="plot.png"):
    # plot the training loss and accuracy
    epochs = len(H.history["loss"])
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    if save_plot is not None:
        plt.savefig(save_plot)
    else:
        plt.show()


def classification_show(images, predictions, label_names, save_plot="plot.png"):
    """

    :param images:
    :param predictions:
    :param label_names:
    :param save_plot:
    :return:
    """
    assert len(images) == len(predictions)

    plt.figure(figsize=(10, 10))
    for i, img in enumerate(images):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        prediction_label_index = predictions[i]
        predict_label = label_names[prediction_label_index]
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(rgb_img)
        plt.xlabel(predict_label)
    if save_plot is not None:
        plt.savefig(save_plot)
    else:
        plt.show()
