from keras.models import Model, Sequential, load_model
from keras.layers import Input, LSTM, Dense, TimeDistributed, Lambda, Dropout
from keras import backend as K

class LSTM_model():

    def __init__(self, num_features=2048, hidden_units=256, dense_units=256, reg=1e-1, dropout_rate=1e-1, seq_length=16, num_classes=3):
            # hidden_units: dimension of cell
            # dense_units: number of neurons in fully connected layer above LSTM
            # reg: regularization for LSTM and dense layer
            # - currently adding L2 regularization for RNN connections, and for inputs to dense layer

            model = Sequential()

            # return_sequences flag sets whether hidden state returned for each time step
            # NOTE: set return_sequences=True if using TimeDistributed, else False


            # LSTM layer (dropout)
            model.add(Dropout(dropout_rate, input_shape=(seq_length, num_features)))  # input to LSTM
            model.add(LSTM(hidden_units, return_sequences=True))

            # --- AVERAGE LSTM OUTPUTS --- #

            # dropout between LSTM and softmax
            model.add(TimeDistributed(Dropout(dropout_rate)))

            # commenting out additional FC layer for now
            # model.add(TimeDistributed(Dense(dense_units)))

            # apply softmax
            model.add(TimeDistributed(Dense(num_classes, activation="softmax")))

            # average outputs
            average_layer = Lambda(function=lambda x: K.mean(x, axis=1))
            model.add(average_layer)

            self.model = model
            # --- ONLY TAKE LAST LSTM OUTPUT --- #
            # model.add(Dense(dense_units))
            # model.add(Dense(num_classes, activation="softmax"))

