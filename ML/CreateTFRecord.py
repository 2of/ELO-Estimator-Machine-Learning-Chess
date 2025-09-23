from pathlib import Path
import sys
# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ingest.ParseGame import MultiFileLoaderWrapper
from ML.MoveHash import *
import pandas as pd
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

class RecordBuilder:
    def __init__(self, fileDirectory, output_dir="./data/tfrecords", shard_size=500):
        self.wrapper = MultiFileLoaderWrapper(fileDirectory)
        self.currentBatch = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.shard_index = 0

        self.hashmap = MoveHashHandler("./data/MovesHash/Moveshash.pd")
        self.hashmap.load_map()

    def get_next_batch(self, n=50):
        """Fetch next batch of n games"""
        batch = self.wrapper.get_n_games(n)
        if batch:
            self.currentBatch = batch
            return batch
        return []

    def replace_tokens(self, row: str):
        """Replace moves in a game with integer IDs from the hashmap"""
        res = []
        tokens = row[0].split()  # space-separated moves
        for token in tokens:
            value = self.hashmap.get(token)
            if value is not None:
                res.append(int(value))
        return res

    def split_record(self, row):
        """Return (white_elo, black_elo, moves)"""
        return row[0], row[1], row[2:]

    def get_augmented_row(self, row):
        white, black, moves = self.split_record(row)
        moves_int = self.replace_tokens(moves)
        return float(white), float(black), moves_int

    def decompose_batch(self, batch, max_len=128):
        """Convert batch to padded sequences and labels"""
        X_list, y_list = [], []
        for row in batch:
            white, black, moves = self.get_augmented_row(row)
            X_list.append(moves)
            y_list.append([white, black])

        X_padded = pad_sequences(X_list, maxlen=max_len, padding='post', truncating='post', value=0)
        X_tensor = tf.convert_to_tensor(X_padded, dtype=tf.int32)
        y_tensor = tf.convert_to_tensor(y_list, dtype=tf.float32)
        return X_tensor, y_tensor


    def _serialize_example(self, X_row, y_row):
        """Convert one sample to TFRecord Example"""
        feature = {
            "moves": tf.train.Feature(int64_list=tf.train.Int64List(value=X_row)),
            "white_elo": tf.train.Feature(float_list=tf.train.FloatList(value=[y_row[0]])),
            "black_elo": tf.train.Feature(float_list=tf.train.FloatList(value=[y_row[1]]))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

    def write_tfrecord_shard(self, X_tensor, y_tensor):
        """Write one shard of TFRecords"""
        shard_path = self.output_dir / f"games_{self.shard_index:04d}.tfrecord"
        with tf.io.TFRecordWriter(str(shard_path)) as writer:
            for i in range(X_tensor.shape[0]):
                example = self._serialize_example(X_tensor[i].numpy(), y_tensor[i].numpy())
                writer.write(example)
        print(f"✅ Wrote TFRecord shard: {shard_path}")
        self.shard_index += 1

    def build_tfrecords(self, batch_size=500, max_len=256):
        """Main loop: stream batches, convert, and write TFRecords"""
        total_games = 0
        while True:
            batch = self.get_next_batch(batch_size)
            if not batch:
                break

            X_tensor, y_tensor = self.decompose_batch(batch, max_len=max_len)
            self.write_tfrecord_shard(X_tensor, y_tensor)

            total_games += len(batch)
            print(f"Processed {len(batch)} games, total so far: {total_games}")

        print(f"All done! Total games processed: {total_games}")


if __name__ == "__main__":
    rb = RecordBuilder(output_dir="/volumes/bck/CHESS/TFRecords", fileDirectory="/volumes/bck/LICHESS")
    rb.build_tfrecords(batch_size=500000, max_len=128)