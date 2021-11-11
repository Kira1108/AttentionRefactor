import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Bidirectional

class Encoder(tf.keras.layers.Layer):
    """Encode a sequence of length T, vocab size V, embedding dimension D and hidden states M"""
    def __init__(self, T, V, D, M, **kwargs):
        super().__init__(**kwargs)
        self.embedding = Embedding(V, D, input_length = T)
        self.lstm = Bidirectional(LSTM(M, return_sequences = True))

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm(x)
        return x

def test_encoder(T = 10, V = 2000, D = 200, M = 15, N = 128):
    data = np.random.random((N, T))
    encoder = Encoder(T, V, D, M)
    assert encoder(data).shape == (N, T, 2 * M)
    print("Encoder passed test")