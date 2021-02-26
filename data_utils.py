"""
Functions for preparing, processing, and manipulating video data
This file was adapted from https://github.com/harvitronix/five-video-classification-methods.
Contains functionality for preprocessing video data files and generating features from frames

"""

from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.utils import to_categorical

import numpy as np
import cv2 as cv
import random
import csv
import sys

import time
import os
import os.path

import threading
import operator

class DataSet():

    def __init__(self,
                 cnn_model,
                 seq_length=16,
                 seq_filepath=os.path.join('data', 'sequences'),
                 cnnmod_inputdim=(299, 299),
                 ):
        """ Constructor for DataSet() class
        cnn_model = Keras CNN Model, used for generating sequences from the video input data
        seq_length = (int) the number of frames to downsample the video to
        seq_filepath = filepath of where the sequence data should be saved/stored
        cnnmod_inputdim = input dimension of image for the CNN model e.g. (299, 299) for InceptionV3
        """

        # InceptionV3 model will be used to extract features from frames
        self.cnn_model = cnn_model

        # length of each video in the dataset (default downsampled to 16 frames)
        self.seq_length = seq_length

        # directory containing saved .npy sequences -- must create a data/sequences directory in the
        # current directory
        self.seq_filepath = seq_filepath

        # obtain the dataset info from data_file.csv (outputted from dataprep_split.py)
        self.data = self.get_data()

        # obtain info on the classes from self.data
        self.classes = self.get_classes()

        # the required input dimensions of the CNN model
        self.cnnmod_inputdim = cnnmod_inputdim  # for potentially changing image sizes down the road


    @staticmethod
    def get_data():
        """
        Load our data from file. Must be contained in a data folder in the same directory and be
        named data_file.csv --> data/data_file.csv
        :return: info (columns) of input data csv file
        """
        with open(os.path.join('data', 'data_file.csv'), 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)
        return data


    def get_classes(self):
        """
        This function creates a list of classes from data_file.csv, which is outputted from dataprep_split.py
        :return: sorted list of class labels
        """
        classes = []
        for item in self.data:
            if item[1] not in classes:
                classes.append(item[1])

        classes = sorted(classes)  # class labels sorted in alphabetical order

        return classes


    def get_class_one_hot(self, class_label):
        """
        This function does a one-hot encoding corresponding to the class label.
        :return: one-hot vector corresponding to the input class label
        """
        label = self.classes.index(class_label)

        label_hot = to_categorical(label, len(self.classes))

        assert len(label_hot) == len(self.classes)

        return label_hot


    def split_dataset(self):
        """
        Function to explicitly split out lists of train/dev/test data
        :return: train/dev/test data lists
        """
        train = []
        dev = []
        test = []
        for item in self.data:
            if item[0] == 'train':
                train.append(item)
            elif item[0] == 'dev':
                dev.append(item)
            else:
                test.append(item)
        return train, dev, test


    def get_frames_for_sample(self, sample):
        """
        This function, used in extract_seq_features(), obtains a list of
        frames from a given sample video. Each frame is a 3-d matrix.
        The raw video .avi files must be saved in a VIDEO_RGB folder followed by the class subfolder
        e.g. "VIDEO_RGB/backhand/p1_backhand_s1.avi"

        sample := a row in the data_file.csv of format
        [type of data (train/dev/test), class, video_filename without .avi]

        :return: list of downsampled frames in RGB-format
        """

        vidfilepath = os.path.join('VIDEO_RGB', sample[1], sample[2] + '.avi')

        # process the video and extract the sequence of frames
        vidcap = cv.VideoCapture(vidfilepath)

        frames = []

        # extract frames
        while True:
            ret, frame = vidcap.read()
            if not ret:
                break

            # data augmentation on frames (for additional data)
            # frame_vertflip = cv.flip(frame, 0)
            # frame_horzflip = cv.flip(frame, 1)
            # frame_bothaxesflip = cv.flip(frame, -1)
            # frame_rotate = cv.transpose(frame)

            # openCV package defaults to BGR ordering --> switch to RGB
            img_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            frames.append(img_RGB)

        # downsample
        if self.seq_length < len(frames):
            skip = len(frames) // self.seq_length
            frames = [frames[i] for i in range(0, len(frames), skip)]
            frames = frames[:self.seq_length]

        return frames


    def extract_seq_features(self, sample):
        """
        This function, used in get_extracted_sequence(), returns
        a sequence of CNN-generated features for frames and saves to .npy file

        :return: sequence of CNN-generated features from the input frame pixels
        """

        savepath = os.path.join('data', 'sequences', sample[1], sample[2] + '-' + str(self.seq_length) + \
        '-features')

        # sample the sequence of frames from the video
        frames = self.get_frames_for_sample(sample)

        # preprocess frames and feed into the CNN model
        sequence = []
        for img in frames:
            # reshape raw frame to fit CNN model input (299,299,3)
            img = cv.resize(img, self.cnnmod_inputdim, interpolation = cv.INTER_AREA)

            # reshape frame for InceptionV3 model input
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x) # standardize RGB frames and preprocessing

            # generate features and save them
            features = self.cnn_model.predict(x)
            sequence.append(features[0])  # get the relevant 1-d np array (features is 2-d)

        np.save(savepath, sequence)

        return sequence


    def get_extracted_sequence(self, sample):
        """
        This function is used in frame_generator(). Returns a sequence
        (I believe this is a list) from a .npy file stored on disc
        or creates it on the fly if not and saves as .npy for future use.

        Each sequence is a list, with each element containing the feature
        vector for each frame.
        """

        filename = sample[2]

        path = os.path.join(self.seq_filepath, sample[1], filename + '-' + str(self.seq_length) + \
            '-features.npy')

        # return saved numpy sequence
        if os.path.isfile(path):
            return np.load(path)

        # else we generate the numpy sequence now (saved on disc for future use)
        else:
            return self.extract_seq_features(sample)


    def get_frames_by_filename(self, filename):

        sample = None
        for row in self.data:
            if row[2] == filename:
                sample = row
                break
        if sample is None:
            raise ValueError("Could not find sample")

        sequence = self.get_extracted_sequence(sample)

        return sequence


    def print_class_from_prediction(self, predictions):

        label_predictions = {}

        for i, label in enumerate(self.classes):
            label_predictions[label] = predictions[i]

        sorted_lps = sorted(label_predictions.items(), key=operator.itemgetter(1), reverse=True)

        for i, class_prediction in enumerate(sorted_lps):
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))


    @threadsafe_generator
    def frame_generator(self, batch_size, train_validate):
        """
        This function creates a generator that we will use during training.
        """

        # random.seed(1)  # for reproducibility in experiments?

        # not using actual test data, using validation data during training
        train, validation, _ = self.split_dataset()
        data = train if train_validate == 'train' else validation

        print("Creating %s generator with %d samples.\n" % (train_validate, len(data)))

        if train_validate == 'train':
            while 1:
                X, y = [], []

                # generate samples for batch (of size batch_size)
                for _ in range(batch_size):

                    sequence = None

                    # randomly pick a datapoint (what if we have already picked point in batch?)
                    sample = random.choice(data)

                    sequence = self.get_extracted_sequence(sample)

                    if sequence is None:
                        raise ValueError("Unable to find sequence!")

                    X.append(sequence)

                    class_label = sample[1]  # from csv line
                    y.append(self.get_class_one_hot(class_label))

                # yield batches as necessary to fit_generator() fxn
                yield np.array(X), np.array(y)
        else:
            while 1:
                X, y = [], []
                for i in range(len(data)):

                    sequence = None

                    sample = data[i]

                    sequence = self.get_extracted_sequence(sample)

                    if sequence is None:
                        raise ValueError("Unable to find sequence!")

                    X.append(sequence)

                    class_label = sample[1]  # from csv line
                    y.append(self.get_class_one_hot(class_label))

                # print "yielding entire validation set"
                yield np.array(X), np.array(y)


    def generate_data(self, train_validate_test):
        """
        This function generates desired training data
        """
        train, validation, test = self.split_dataset()
        if train_validate_test == 'train':
            data = train
        elif train_validate_test == 'validation':
            data = validation
        elif train_validate_test == 'test':
            data = test

        X, y = [], []

        # loop over list of validation samples, and create sequences
        for sample in data:

            sequence = None

            sequence = self.get_extracted_sequence(sample)

            if sequence is None:
                raise ValueError("Unable to find sequence!")

            X.append(sequence)

            class_label = sample[1]  # from csv line
            y.append(self.get_class_one_hot(class_label))

        # print "yielding entire validation set"
        return np.array(X), np.array(y)

    def get_extracted_sequences(self, train_validate_test):
        """
        This function gets extracted sequences saved as npy
        """

        train, validation, test = self.split_dataset()
        if train_validate_test == 'train':
            data = train
        elif train_validate_test == 'validation':
            data = validation
        elif train_validate_test == 'test':
            data = test

        X, y = [], []
        # loop over list of validation samples, and create sequences
        for sample in data:
            sequence = None
            sequence_path = os.path.join('data', 'sequences', sample[1], sample[2] + '-' + str(self.seq_length) + '-features.npy')
            sequence = np.load(sequence_path)

            if sequence is None:
                raise ValueError("Unable to find sequence!")

            X.append(sequence)

            class_label = sample[1]  # from csv line
            y.append(self.get_class_one_hot(class_label))

        # print "yielding entire validation set"
        return np.array(X), np.array(y)