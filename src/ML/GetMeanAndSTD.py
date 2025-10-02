import numpy as np
import tensorflow as tf
from helpers.TFRecordLoader import RecordLoadWrapper
from models_templates.Models import *

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "./data/tfrecordsamples/small"
BATCH_SIZE = 400
MAX_LEN = 256
SHUFFLE_BUFFER = 10000
EPOCHS = 20

#-----------
# Loaders
#-----------

train_loader = RecordLoadWrapper(
    recorddir=DATA_DIR + "/train",
    batchsize=BATCH_SIZE,
    shuffle_buffer=SHUFFLE_BUFFER,
    max_len=MAX_LEN,
)

val_loader = RecordLoadWrapper(
    recorddir=DATA_DIR + "/val",
    batchsize=BATCH_SIZE,
    shuffle_buffer=SHUFFLE_BUFFER,
    max_len=MAX_LEN,
)

test_loader = RecordLoadWrapper(
    recorddir=DATA_DIR + "/test",
    batchsize=BATCH_SIZE,
    shuffle_buffer=SHUFFLE_BUFFER,
    max_len=MAX_LEN,
)

# -----------------------------
# Compute mean & std of Elo labels
# -----------------------------
all_labels = []

train_loader.reset()
while True:
    batch = train_loader.get_next_batch()
    if batch is None:
        break
    _, labels = batch   # labels shape: (batch, 2) -> [white_elo, black_elo]
    all_labels.append(labels.numpy())

all_labels = np.concatenate(all_labels, axis=0)  # shape: (N, 2)

elo_mean = np.mean(all_labels, axis=0)  # shape (2,)
elo_std = np.std(all_labels, axis=0)    # shape (2,)

print("Elo mean:", elo_mean)
print("Elo std:", elo_std)

# Convert to TF constants for training use
elo_mean_tf = tf.constant(elo_mean, dtype=tf.float32)
elo_std_tf = tf.constant(elo_std, dtype=tf.float32)



