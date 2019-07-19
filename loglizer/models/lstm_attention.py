import keras
from keras import layers
import functools


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = int(inputs.shape[1])
    input_dim = int(inputs.shape[2])
    a = layers.Permute((2, 1))(inputs)
    a = layers.Reshape((input_dim, time_steps))(a) # this line is not useful. It's just to know which dimension is what.
    a = layers.Dense(time_steps, activation='softmax')(a)
    a_probs = layers.Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = layers.multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


class LSTMAttention(object):
    def __init__(self, g, h, L, alpha, batch_size, sym_count, lr=0.001):
        self.g, self.h, self.L, self.alpha, self.batch_size, self.sym_count, self.lr = g, h, L, alpha, batch_size, sym_count, lr

        self.input = keras.Input((self.h, self.sym_count))
        # self.attention = attention_3d_block(self.input)
        # self.lstm = self.attention
        self.lstm = self.input
        for _ in range(self.L - 1):
            self.lstm = layers.LSTM(self.alpha, return_sequences=True, dropout=0.5)(self.lstm)
        # self.lstm = layers.LSTM(self.alpha)(self.lstm)
        self.lstm = layers.LSTM(self.alpha, return_sequences=True)(self.lstm)
        self.lstm = layers.Reshape((h, alpha))(self.lstm)
        self.attention = attention_3d_block(self.lstm)
        self.flatten = layers.Flatten()(self.attention)
        # self.dense = layers.Dense(self.sym_count, activation='softmax')(self.lstm)
        self.dense = layers.Dense(self.sym_count, activation='softmax')(self.flatten)

        self.model = keras.Model(inputs=[self.input], outputs=self.dense)
        top_g_categorical_accuracy = functools.partial(keras.metrics.top_k_categorical_accuracy, k=self.g)
        top_g_categorical_accuracy.__name__ = 'top_g_categorical_accuracy'
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adam(),
                           metrics=[keras.metrics.categorical_accuracy, top_g_categorical_accuracy])

        self.model.summary()


if __name__ == '__main__':
    model = LSTMAttention(9, 10, 2, 64, 128, 29)
    pass