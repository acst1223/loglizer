# loglizer

A deep learning log anomaly detection tool extended from [logpai/loglizer](https://github.com/logpai/loglizer).



### Models

This tool contains supervised detection models as well as unsupervised detection models.

#### Supervised detection models

For supervised detection models, anomaly labels are necessary.

- **CNN**: A Convolutional Neural Network with log event lists as its input. It needs anomaly labels as its reference.

#### Unsupervised detection models

Unsupervised detection models do not need anomaly labels. They learn only from log event sequences and can be used to detect new sequences after training.

- **LSTM**: A Long-Short Term Memory network. It applies sliding windows on input event sequences and tries to predict next events. If the actual next event is outside top `g` predictions, the sequence that the event belongs to is identified as an anomaly.
- **LSTM-Attention**: A LSTM network with attention mechanism.
- **LSTM-Attention-Count-Vector**: A LSTM-Attention network with event count vectors as additional input.
- **VAE-LSTM**: A Variational Auto-Encoder with LSTM in its encoder and decoder. It applies sliding windows on input event sequences and tries to reconstruct the input window. The log-likelihood for the original input window is computed, and a threshold is determined automatically to judge whether a window is anomalous or not. An event sequence is identified as an anomaly if at least one of its windows is anomalous.

