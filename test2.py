import keras
from keras import layers
import numpy as np


inputs = keras.Input((3, 3))
outputs = layers.Lambda(lambda x: inputs[:, -1])(inputs)
model = keras.Model(inputs=inputs, outputs=outputs)

prediction = model.predict(np.arange(27).reshape((3, 3, 3)))
print(prediction)
