import tensorflow as tf

def parse_line(line):
    line = tf.strings.strip(line)

    if tf.equal(tf.strings.length(line), 0):
        return tf.constant(-1), tf.constant(-1), tf.constant([], dtype=tf.string)

    parts = tf.strings.split(line, sep=",", maxsplit=2)

    if tf.shape(parts)[0] < 3:
        return tf.constant(-1), tf.constant(-1), tf.constant([], dtype=tf.string)
    white_elo = tf.strings.to_number(parts[0], out_type=tf.int32)
    black_elo = tf.strings.to_number(parts[1], out_type=tf.int32)
    moves_str = parts[2]
    moves = tf.strings.split(moves_str, sep=" ")
    return white_elo, black_elo, moves

def load_dataset(file_path, batch_size=32, max_moves=200):
    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.filter(lambda x: tf.strings.length(tf.strings.strip(x)) > 0)  # drop blanks
    dataset = dataset.map(parse_line, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda w, b, m: (
        w,
        b,
        tf.concat([m, tf.fill([tf.maximum(0, max_moves - tf.shape(m)[0])], "")], axis=0)[:max_moves]
    ))

    dataset = dataset.batch(batch_size)
    return dataset

# ---- TFRecord helpers ----
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

def _bytes_list_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.encode("utf-8") for v in values]))

def serialize_example(white_elo, black_elo, moves):
    feature = {
        "white_elo": _int64_feature(white_elo),
        "black_elo": _int64_feature(black_elo),
        "moves": _bytes_list_feature(moves),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_tfrecords(out_path, white_tensor, black_tensor, moves_tensor):
    with tf.io.TFRecordWriter(out_path) as writer:
        for w, b, m in zip(white_tensor.numpy(), black_tensor.numpy(), moves_tensor.numpy()):
            moves_list = [x.decode("utf-8") for x in m if x]  # drop padding blanks
            example = serialize_example(w, b, moves_list)
            writer.write(example)

# ---- Main ----
if __name__ == "__main__":
    file_path = "../data/txt/reduced_games_1.txt"
    out_path = "../data/games.tfrecord"
    max_moves = 200

    ds = load_dataset(file_path, batch_size=32, max_moves=max_moves)

    all_white_elos = []
    all_black_elos = []
    all_moves = []
    total_games = 0

    for white_elo, black_elo, moves in ds:
        all_white_elos.append(white_elo)
        all_black_elos.append(black_elo)
        all_moves.append(moves)
        total_games += white_elo.shape[0]

    # Stack into tensors
    white_tensor = tf.concat(all_white_elos, axis=0)
    black_tensor = tf.concat(all_black_elos, axis=0)
    moves_tensor = tf.concat(all_moves, axis=0)

    # Shape as (2, n) and (max_moves, n)
    labels_tensor = tf.stack([white_tensor, black_tensor], axis=0)  # (2, n)
    moves_tensor = tf.transpose(moves_tensor)  # (max_moves, n)

    print(f"✅ Total valid games read: {total_games}")
    print("Labels shape:", labels_tensor.shape)
    print("Moves shape:", moves_tensor.shape)
    tf.print("Moves tensor shape:", tf.shape(moves))
    tf.print("First game's moves:", moves[0])

    # ---- Write TFRecords ----
    write_tfrecords(out_path, white_tensor, black_tensor, moves_tensor)
    print(f"📦 TFRecords written to {out_path}")