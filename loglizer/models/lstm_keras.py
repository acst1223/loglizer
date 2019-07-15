import keras
from keras import layers
import functools


class LSTM(object):
    def __init__(self, g, h, L, alpha, batch_size, sym_count, lr=0.001):
        self.g, self.h, self.L, self.alpha, self.batch_size, self.sym_count, self.lr = g, h, L, alpha, batch_size, sym_count, lr

        self.input = keras.Input((self.h, self.sym_count))
        self.lstm = self.input
        for _ in range(self.L - 1):
            self.lstm = layers.LSTM(self.alpha, return_sequences=True, dropout=0.5)(self.lstm)
        self.lstm = layers.LSTM(self.alpha, dropout=0.5)(self.lstm)
        self.dense = layers.Dense(self.sym_count, activation='softmax')(self.lstm)

        self.model = keras.Model(inputs=[self.input], outputs=self.dense)
        top_g_categorical_accuracy = functools.partial(keras.metrics.top_k_categorical_accuracy, k=self.g)
        top_g_categorical_accuracy.__name__ = 'top_g_categorical_accuracy'
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adam(),
                           metrics=[keras.metrics.categorical_accuracy, top_g_categorical_accuracy])

        self.model.summary()
