# import the necessary packages
import os

import h5py


class HDF5DatasetWriter:
    def __init__(self, dims, label_dims, output_path, data_key="images", buf_size=1000, overwrite=False, **kwargs):
        """
        :param dims: (tuple) dims of the data to write first dim need to be length of data
        :param label_dims: (int) dimention of labels
        :param output_path: (string) path to save the h5 file
        :param data_key: (string) Key to get images
        :param buf_size:(int) Buffer size before save into the file
        :param overwrite: (bool) Overwrite if file existe
        :param kwargs: (dict)
            All dtype can be found here: http://docs.h5py.org/en/stable/faq.html#faq
            * image_dtype: (string) dtype of images default: "float"
            * labels_dtype: (string) dtype of labels default: "int"

        """
        # check to see if the output path exists, and if so, raise
        # an exception
        if os.path.exists(output_path):
            if not overwrite:
                raise ValueError("The supplied `outputPath` already "
                                 "exists and cannot be overwritten. Manually delete "
                                 "the file before continuing.", output_path)
            else:
                os.remove(output_path)

        # open the HDF5 database for writing and create two datasets:
        # one to store the images/features and another to store the
        # class labels
        self.db = h5py.File(output_path, "w")
        dtype = kwargs.get("image_dtype", "float")
        self.data = self.db.create_dataset(data_key, dims,
                                           dtype=dtype)
        dtype = kwargs.get("labels_dtype", "int")
        self.labels = self.db.create_dataset("labels", (dims[0], label_dims),
                                             dtype=dtype)

        # store the buffer size, then initialize the buffer itself
        # along with the index into the datasets
        self.bufSize = buf_size
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, rows, labels):
        # add the rows and labels to the buffer
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        # check to see if the buffer needs to be flushed to disk
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self):
        # write the buffers to disk then reset the buffer
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def storeClassLabels(self, classLabels):
        # create a dataset to store the actual class label names,
        # then store the class labels
        dt = h5py.special_dtype(vlen=str)  # `vlen=unicode` for Py2.7
        labelSet = self.db.create_dataset("label_names",
                                          (len(classLabels),), dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        # check to see if there are any other entries in the buffer
        # that need to be flushed to disk
        if len(self.buffer["data"]) > 0:
            self.flush()

        # close the dataset
        self.db.close()
