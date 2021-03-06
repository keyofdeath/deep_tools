# import the necessary packages
from sklearn.feature_extraction.image import extract_patches_2d

from deep_tools.dataset.preprocessing import Preprocessing


class PatchPreprocessor(Preprocessing):
    def __init__(self, width, height):
        # store the target width and height of the image
        self.width = width
        self.height = height

    def preprocess(self, image):
        # extract a random crop from the image with the target width
        # and height
        return extract_patches_2d(image, (self.height, self.width),
                                  max_patches=1)[0]
