from tensorflow.keras.applications.xception import Xception, preprocess_input
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense
from data_utils import DataSet

import numpy as np
import os, sys
import time


def load_cnn_model():
    """
    Loads the Xception pre-trained CNN model
    :return: Xception model
    """
    base_model = Xception(weights='imagenet', include_top=True)

    # get the feature outputs of second-to-last layer (final FC layer)
    outputs = base_model.get_layer('avg_pool').output

    cnn_model = Model(inputs=base_model.input, outputs=outputs)

    return cnn_model

if __name__ == "__main__":

    cnn_model = load_cnn_model()

    seq_length = 16 # sequence length of frames to downsample each video to
    dataset = DataSet(cnn_model)

    # generate Xception features and time it
    currtime = time.time()

    for ind, sample in enumerate(dataset.data):
        # save the sequences of frame features to npy files for eventual model training
        path = os.path.join('data', 'sequences', sample[1], sample[2] + '-' + str(seq_length) + '-Xception_features.npy')

        if os.path.isfile(path):
            print(sample)
            print("Sequence: {} already exists".format(ind))
        else:
            print(sample)
            print("Generating and saving sequence: {}".format(ind))
            sequence = dataset.extract_seq_features(sample, Xception=True)

    print("Time Elapsed: {}".format(time.time() - currtime))