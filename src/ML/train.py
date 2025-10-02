import tensorflow as tf
from helpers.TFRecordLoader import RecordLoadWrapper

from models_templates.Models import *
from helpers import SaveHelper
# -----------------------------
# Config
# -----------------------------
DATA_DIR = "./data/tfrecordsamples/small"
BATCH_SIZE = 400
MAX_LEN = 256
SHUFFLE_BUFFER = 10000
EPOCHS = 20
DATA_STD = [400.12628, 400.09366]
DATA_MEAN = [1667.8077, 1667.8461]



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

models = [SimpleNNModel(), Attention_EloModel(),LSTMModel(),LSTM_Attention_EloModel()]



#--------
# Training time
#--------


def normalize_labels(labels):
    return (labels - DATA_MEAN) / DATA_STD

def denormalize_preds(preds):
    return preds * DATA_STD + DATA_MEAN






@tf.function
def train_step(moves, labels):
    labels_norm = normalize_labels(labels)
    with tf.GradientTape() as tape:
        preds_norm = model(moves, training=True)
        loss = loss_fn(labels_norm, preds_norm)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss_metric.update_state(loss)

    # Debugging inside tf.function
    # tf.print("Batch loss:", loss, "labels[0]:", labels[0], "preds[0]:", predictions[0])

@tf.function
def val_step(moves, labels):
    labels_norm = normalize_labels(labels)
    preds_norm = model(moves, training=False)
    v_loss = loss_fn(labels_norm, preds_norm)
    val_loss_metric.update_state(v_loss)


@tf.function
def test_step(moves, labels):
    labels_norm = normalize_labels(labels)
    preds_norm = model(moves, training=False)
    t_loss = loss_fn(labels_norm, preds_norm)
    test_loss_metric.update_state(t_loss)


for model in models:
    print("NOW ON MODEL", model)
    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

        # Reset metrics
        train_loss_metric.reset_state()
        val_loss_metric.reset_state()

        # ---- Training ----
        print("Training...")
        train_loader.reset()
        batch_idx = 0
        while True:
            batch = train_loader.get_next_batch()
            if batch is None:
                break
            moves, labels = batch
            train_step(moves, labels)
            batch_idx += 1

            # if batch_idx % 5 == 0:  # every 5 batches
            #     print(f"  [Train] Batch {batch_idx} | Current batch loss: {train_loss_metric.result().numpy():.4f}")

        print(f"  [Train] Epoch {epoch+1} average loss: {train_loss_metric.result().numpy():.4f}")

        # ---- Validation ----
        print("Validating...")
        val_loader.reset()
        while True:
            batch = val_loader.get_next_batch()
            if batch is None:
                break
            moves, labels = batch
            val_step(moves, labels)

        print(f"  [Val]   Epoch {epoch+1} average loss: {val_loss_metric.result().numpy():.4f}")


    # ---- Final Test Evaluation ----
    print("\n=== Final Test Evaluation ===")
    test_loss_metric.reset_state()
    test_loader.reset()
    batch_idx = 0
    while True:
        batch = test_loader.get_next_batch()
        if batch is None:
            break
        moves, labels = batch
        test_step(moves, labels)
        batch_idx += 1

    print(f"[Test] Final average loss: {test_loss_metric.result().numpy():.4f}")

    # ---- Save Models ----
    print("\n=== Saving final model ===")
    SaveHelper.save_regular(model, model_name=f"{model}_MODEL")
    SaveHelper.save_for_js(model, model_name=f"{model}_MODEL_JS")