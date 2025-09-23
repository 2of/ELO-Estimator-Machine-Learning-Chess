import tensorflow as tf
from helpers.TFRecordLoader import RecordLoadWrapper

from ML.models_templates.Models import *

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "./data/tfrecordsamples/small"
BATCH_SIZE = 400
MAX_LEN = 256
SHUFFLE_BUFFER = 10000
EPOCHS = 2

#-----------
# Loaders
#-----------

test_loader = RecordLoadWrapper(
    recorddir=DATA_DIR+"/test",
    batchsize=BATCH_SIZE,
    shuffle_buffer=SHUFFLE_BUFFER,
    max_len=MAX_LEN,
)


train_loader = RecordLoadWrapper(
    recorddir=DATA_DIR+"/train",
    batchsize=BATCH_SIZE,
    shuffle_buffer=SHUFFLE_BUFFER,
    max_len=MAX_LEN,
)


val_loader = RecordLoadWrapper(
    recorddir=DATA_DIR+"/val",
    batchsize=BATCH_SIZE,
    shuffle_buffer=SHUFFLE_BUFFER,
    max_len=MAX_LEN,
)


#-----------
# Other Hyperparameters
#-----------


loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
val_loss_metric   = tf.keras.metrics.Mean(name="val_loss")
test_loss_metric  = tf.keras.metrics.Mean(name="test_loss")


model = SimpleNNModel()



#--------
# Training time
#--------


@tf.function
def train_step(moves, labels):
    print(moves.shape)
    print(labels.shape)
    with tf.GradientTape() as tape:
        predictions = model(moves, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss_metric.update_state(loss)

@tf.function
def val_step(moves, labels):
    predictions = model(moves, training=False)
    v_loss = loss_fn(labels, predictions)
    val_loss_metric.update_state(v_loss)

@tf.function
def test_step(moves, labels):
    predictions = model(moves, training=False)
    t_loss = loss_fn(labels, predictions)
    test_loss_metric.update_state(t_loss)






for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
       # ---- Training ----
    train_loader.reset()
    while True:
        batch = train_loader.get_next_batch()
        # print(batch)
        if batch is None:
            break
        moves, labels = batch
        train_step(moves, labels)
