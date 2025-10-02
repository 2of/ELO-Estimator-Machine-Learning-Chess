import tensorflow as tf

class SimpleNNModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=6400, output_dim=64)  # vocab size
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu")
        self.output_layer = tf.keras.layers.Dense(2, activation="linear")  # white_elo, black_elo
    def __str__(self):
        return("SIMPLENN")
    def call(self, inputs, training=False):
        # print("Input shape:", inputs.shape)
        x = self.embedding(inputs)
        # print("After embedding:", x.shape)
        x = self.flatten(x)
        # print("After flatten:", x.shape)
        x = self.dense1(x)
        # print("After dense1:", x.shape)
        x = self.dense2(x)
        # print("After dense2:", x.shape)
        out = self.output_layer(x)
        # print("Output shape:", out.shape)
        return out


class Attention_EloModel(tf.keras.Model):
    def __init__(self, vocab_size=6400, embed_dim=64, num_heads=4, ff_dim=128):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(ff_dim, activation="relu")
        ])
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        self.output_layer = tf.keras.layers.Dense(2, activation="linear")
    def __str__(self):
        return("Attention_EloModel")
    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        attn_out = self.attention(x, x)
        x = x + attn_out  # residual connection
        x = self.ff(x)
        x = self.pool(x)
        out = self.output_layer(x)
        return out

class LSTMModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=6400, output_dim=64)
        self.lstm = tf.keras.layers.LSTM(128, return_sequences=False)  # summarize the sequence
        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.output_layer = tf.keras.layers.Dense(2, activation="linear")
    def __str__(self):
        return("LSTMModel")
    def call(self, inputs, training=False):
        print("Input shape:", inputs.shape)
        x = self.embedding(inputs)
        print("After embedding:", x.shape)
        x = self.lstm(x)
        print("After LSTM:", x.shape)
        x = self.dense1(x)
        print("After dense1:", x.shape)
        out = self.output_layer(x)
        print("Output shape:", out.shape)
        return out
    
class LSTM_Attention_EloModel(tf.keras.Model):
    def __init__(self, vocab_size=6400, embed_dim=64, lstm_units=128, num_heads=4, ff_dim=128):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True))
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=lstm_units*2)
        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(ff_dim, activation="relu")
        ])
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        self.output_layer = tf.keras.layers.Dense(2, activation="linear")
    def __str__(self):
        return("LSTM_Attention_EloModel")
    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.lstm(x)
        attn_out = self.attention(x, x)
        x = x + attn_out  # residual
        x = self.ff(x)
        x = self.pool(x)
        out = self.output_layer(x)
        return out
if __name__ =="__main__":
    # Instantiate the model
    model = LSTM_Attention_EloModel()

    # Dummy input: batch of 8, sequence length 10 (values must be < 6400)
    dummy_input = tf.random.uniform(shape=(8, 256), minval=0, maxval=6400, dtype=tf.int32)

    # Forward pass
    output = model(dummy_input)

    print("Final output:", output)