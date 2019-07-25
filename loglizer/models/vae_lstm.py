import keras
from keras import layers, losses
import keras.backend as K


class VAELSTM(object):
    def __init__(self, h, sym_count, batch_size, z_dim=20, alpha=64):
        self.h, self.sym_count, self.batch_size = h, sym_count, batch_size

        inputs = keras.Input((self.h, self.sym_count))
        q_lstm = layers.LSTM(alpha)(inputs)

        z_mean = layers.Dense(z_dim)(q_lstm)
        z_log_std = layers.Dense(z_dim)(q_lstm)

        def sampling(args):
            z_mean, z_log_std = args
            epsilon = K.random_normal(shape=(batch_size, z_dim))
            return z_mean + z_log_std * epsilon

        z = layers.Lambda(sampling, output_shape=(z_dim,))([z_mean, z_log_std])

        p_repeat = layers.RepeatVector(self.h)(z)
        p_lstm = layers.LSTM(alpha, return_sequences=True)(p_repeat)
        p_lstm = layers.LSTM(self.sym_count, return_sequences=True)(p_lstm)
        p_output = layers.Softmax()(p_lstm)
        # p_output = p_lstm

        nll = layers.Lambda(lambda args: K.mean(K.categorical_crossentropy(args[0], args[1]), axis=1))([inputs, p_output])
        # nll = layers.Lambda(lambda args: K.mean(losses.mse(args[0], args[1]), axis=1))(
        #     [inputs, p_output])

        def x_loss(x_true, x_pred):
            # return K.mean(losses.mse(x_true, x_pred), axis=1)
            return K.mean(losses.categorical_crossentropy(x_true, x_pred), axis=1)

        def kl_loss(x_true, x_pred):
            return -0.5 * K.mean(1 + z_log_std - K.square(z_mean) - K.exp(z_log_std))

        def vae_loss(x_true, x_pred):
            return 300 * x_loss(x_true, x_pred) + kl_loss(x_true, x_pred)

        self.model = keras.Model(inputs=[inputs], outputs=p_output)
        self.model.compile(loss=vae_loss, optimizer=keras.optimizers.Adam(),
                           metrics=[x_loss, kl_loss])
        self.model.summary()

        self.nll_model = keras.Model(inputs=[inputs], outputs=nll)

