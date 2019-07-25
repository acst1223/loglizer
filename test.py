import keras
from keras import layers, losses
import keras.backend as K
import numpy as np


inputs = keras.Input((2,))
output = layers.Lambda(lambda x: K.mean(inputs, axis=1))(inputs)
model = keras.Model(inputs=inputs, outputs=output)
# model.compile(optimizer='adam', loss=losses.mean_squared_error)


print(model.predict(np.array([[1, 2], [4, 5], [7, 8], [11, 12], [13, 14]])))
