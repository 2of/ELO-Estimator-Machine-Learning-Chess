import tensorflow as tf

def _parse_example(example_proto, max_len=256):
    feature_description = {
        "moves": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "white_elo": tf.io.FixedLenFeature([], tf.float32),
        "black_elo": tf.io.FixedLenFeature([], tf.float32)
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    # Optionally pad moves to max_len
    moves = tf.pad(parsed["moves"], [[0, max_len - tf.shape(parsed["moves"])[0]]])
    return moves, tf.stack([parsed["white_elo"], parsed["black_elo"]])

if __name__ == "__main__":
    tfrecord_path = "./data/tfrecordsamples/games_0000.tfrecord"  # change as needed

    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(lambda x: _parse_example(x))

    for i, (moves, labels) in enumerate(dataset.take(5)):  # preview first 5
        print(f"Sample {i}:")
        print("Moves:", moves.numpy())
        print("Labels (white, black Elo):", labels.numpy())
        print("---")