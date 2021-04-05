# loglizer

A deep learning log anomaly detection tool extended from [logpai/loglizer](https://github.com/logpai/loglizer).



## Models

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



## Installation

Clone all the code to a local directory, e.g. `$DIR`.

Install all the required packages:

```
cd $DIR
pip install -r requirements.txt
```



## Datasets

Now we support two datasets:

- **HDFS** is a dataset collected from Amazon EC2 platform. It contains 11,175,629 log messages. See [Detecting large-scale system problems by mining console logs](https://dl.acm.org/doi/10.1145/1629575.1629587) for more details.
- **BGL** is a dataset recorded by the BlueGene/L supercomputer system at Lawrence Livermore National Labs (LLNL). It contains 4,747,963 log messages. See [What supercomputers say: A study of five system logs](https://ieeexplore.ieee.org/document/4273008) for more details.

To run experiments on these datasets, you should create a `data` folder in `$DIR`, and download the corresponding dataset into the `data` folder from [Google Drive](https://drive.google.com/drive/folders/1tZNJYSpDZ0IpxeF6YciQ-JwRkYm7qs9z?usp=sharing).

To run experiments on your custom dataset, you should pre-process your dataset first to get two data files:

- An event sequence file that organizes all the events in time order for each block/node, like `HDFS/data_instances.csv` for HDFS dataset.
- An event-content reference file, like `HDFS/col_header.csv` for HDFS dataset.

Also, you should add your dataset in `loglizer/config.py` and define how to load your dataset in scripts you would like to run.



## Running

All scripts are in `scripts` directory.

The corresponding script `$SCRIPT` for each model is:

|            Model            |              Script              |
| :-------------------------: | :------------------------------: |
|             CNN             |             `cnn.py`             |
|            LSTM             |            `lstm.py`             |
|       LSTM-Attention        |       `lstm_attention.py`        |
| LSTM-Attention-Count-Vector | `lstm_attention_count_vector.py` |
|          VAE-LSTM           |          `vae_lstm.py`           |

To run scripts, simply

```
cd $DIR/scripts
python $SCRIPT [ARGUMENTS]
```

A brief introduction to some key arguments:

**Common**

- `--dataset`: Name of the dataset. `HDFS` or `BGL`.
- `--epochs`: Epochs to train.

**Supervised detection models**

- `--log_len`: The length that the model will pad each event sequence to.

**Unsupervised detection models**

- `--batch_size`: Batch size.
- `--g`: In LSTM-based models, if the actual next event is in the top `g` predictions, it is considered normal, otherwise anomalous.
- `--h`: Window size.
- `--L`: Number of LSTM layers in LSTM-based models.
- `--alpha`: Number of memory units in LSTM.
- `--plb`: If a sequence is too short, pad it to this length before applying sliding windows.
- `--checkpoint_name`: Checkpoint name. Models will be loaded from and saved to here.
- `--checkpoint_frequency`: To save the model and do evaluation every this epochs.

