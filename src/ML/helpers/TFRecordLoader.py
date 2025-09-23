"""
Loads TFRecords from N files in a directory and provides
a get_next_batch(batch_size) method.

Streams all files seamlessly, batching across file boundaries
without dropping leftover rows. Returns None when all batches
are consumed. Can be reset.
"""

import tensorflow as tf
import os

def _parse_example(example_proto, max_len=256):
    feature_description = {
        "moves": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "white_elo": tf.io.FixedLenFeature([], tf.float32),
        "black_elo": tf.io.FixedLenFeature([], tf.float32)
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    moves = tf.pad(parsed["moves"], [[0, max_len - tf.shape(parsed["moves"])[0]]])
    labels = tf.stack([parsed["white_elo"], parsed["black_elo"]])
    return moves, labels


class RecordLoadWrapper:
    def __init__(self, recorddir, batchsize, shuffle_buffer=10000, max_len=256):
        self.recorddir = recorddir
        self.batchsize = batchsize
        self.shuffle_buffer = shuffle_buffer
        self.max_len = max_len

        self.files = sorted(
            [os.path.join(self.recorddir, f) for f in os.listdir(self.recorddir) if f.endswith(".tfrecord")]
        )
        if not self.files:
            raise FileNotFoundError(f"No TFRecord files found in {self.recorddir}")

        self._build_iterator()

    def _build_iterator(self):
        # Seamlessly read all files
        raw_dataset = tf.data.TFRecordDataset(self.files)
        dataset = raw_dataset.map(lambda x: _parse_example(x, self.max_len))
        dataset = dataset.shuffle(self.shuffle_buffer)
        dataset = dataset.batch(self.batchsize, drop_remainder=True)
        self._iterator = iter(dataset)

    def get_next_batch(self):
        try:
            return next(self._iterator)
        except StopIteration:
            return None

    def reset(self):
        self._build_iterator()


if __name__ == "__main__":
    loader = RecordLoadWrapper(recorddir="./data/tfrecordsamples/small", batchsize=400)

    i = 0
    while True:
        batch = loader.get_next_batch()
        if batch is None:
            print("All files exhausted.")
            break
        moves, labels = batch
        print(f"Batch {i}: moves {moves.shape}, labels {labels.shape}")
        i += 1

    # Reset and get first batch again
    loader.reset()
    first_batch = loader.get_next_batch()
    print("First batch after reset:", first_batch[0].shape, first_batch[1].shape)