# helpers/save_model.py

import os
import tensorflow as tf
import tensorflowjs as tfjs

DEFAULT_SAVE_DIR = "./saved_models"

def save_regular(model, save_dir=None, model_name="elo_model"):
    """
    Saves a Keras/TensorFlow model in the recommended `.keras` format.
    """
    save_dir = save_dir or DEFAULT_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    if not model_name.endswith(".keras") and not model_name.endswith(".h5"):
        model_name += ".keras"

    path = os.path.join(save_dir, model_name)
    model.save(path)  # Native Keras save
    print(f"[INFO] Model saved in standard Keras format at: {path}")
    return path


def save_for_js(model, save_dir=None, model_name="elo_model"):
    """
    Saves a model in TensorFlow.js format for use in web apps.
    """
    save_dir = save_dir or DEFAULT_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    # Always strip .keras/.h5 extensions if present
    base_name = model_name
    if base_name.endswith(".keras") or base_name.endswith(".h5"):
        base_name = os.path.splitext(base_name)[0]

    path = os.path.join(save_dir, base_name + "_tfjs")
    tfjs.converters.save_keras_model(model, path)

    print(f"[INFO] Model saved for TensorFlow.js at: {path}")
    return path