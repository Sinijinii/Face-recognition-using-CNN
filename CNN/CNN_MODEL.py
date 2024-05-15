from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, TimeDistributed, LSTM, Bidirectional, BatchNormalization, Activation, Lambda
from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import numpy as np

class DeepLearning(object):

    def run_model(self, model, X_tr, y_tr, X_val, y_val):
        # compile model
        model.compile(optimizer = Adam(learning_rate = self.eta, beta_1 = self.beta_1, beta_2 = self.beta_2), loss = 'categorical_crossentropy', metrics = 'accuracy')

        history = model.fit(X_tr, y_tr, epochs = self.epochs, validation_data = (X_tr, y_tr), batch_size = self.batch_size)

        return model, history


    def predict_model(self, model, X_te):
        # compute predicted class and respective probability
        pred_class = np.argmax(model.predict(X_te), axis = 1)
        pred_prob  = np.max(model.predict(X_te), axis = 1)

        return pred_class, pred_prob

    def build_cnn_2d(self, img_height, img_width, img_ch):
        # declare model
        model = Sequential()

        # model.add(Resizing(height = img_height, width = img_width)) # resize

        model.add(Conv2D(input_shape=(img_height, img_width, img_ch), filters=9, kernel_size=3, strides=1, padding="same",
                         kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))

        model.add(Conv2D(filters=18, kernel_size=3, strides=1, padding="same", kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))

        model.add(Conv2D(filters=27, kernel_size=3, strides=1, padding="same", kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))

        model.add(Flatten())  # layers[-6]

        model.add(Dense(256))
        model.add(Dropout(0.5))

        model.add(Dense(256))
        model.add(Dropout(0.5))

        model.add(Dense(self.num_class, activation='softmax'))

        return model


