# import the necessary packages
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential


class ShallowNet:

    @staticmethod
    def name():
        return "shallow_net"

    @staticmethod
    def build(width, height, depth, classes):
        """
        :param width: (int) Input image width
        :param height: (int) Input image height
        :param depth: (int) Input image depth (rgb images have a depth equal 3)
        :param classes: (int) number of classes in the dataset.
        :return: (model) Keras model ShallowNet
        """
        # initialize the model along with the input shape to be
        # "channels last"
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # define the first (and only) CONV => RELU layer
        # 32 filter with kernel k size 3x3
        model.add(Conv2D(32, (3, 3), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
