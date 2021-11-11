import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import ( 
    LSTM, 
    Embedding,
    Dense,
    Concatenate
)

class DecoderHeader(tf.keras.layers.Layer):
    def __init__(self, V, D, T, **kwargs):
        super().__init__(**kwargs)
        self.embedding = Embedding(V, D, input_length = T)

    def call(self, inputs):
        x = self.embedding(inputs)
        return x

class Decoder(tf.keras.layers.Layer):

    def __init__(self, V, M, T = 1, **kwargs):
        super().__init__(**kwargs)
        self.M = M
        self.lstm = LSTM(M, return_state = True)
        self.dense = Dense(V, activation = 'softmax')
        self.concat = Concatenate(axis = -1)

    def call(self, decoder_embedding_inputs, context, state = None):
        
        if not state:
            s = tf.zeros((len(decoder_embedding_inputs),self.M))
            state = [s, s]

        # concat context and embedded vectors
        x = self.concat([decoder_embedding_inputs, context])

        # go through decoder lstm
        o, h, c = self.lstm(x, initial_state = state)

        # go through decoder dense
        output = self.dense(o)

        # return decoder final output and lstm states
        return output, [h,c]