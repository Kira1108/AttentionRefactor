import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Input, 
    LSTM, 
    Embedding,
    Dense,
    Concatenate
)


class DecoderHeader(tf.keras.models.Model):
    '''A simple decoder embedding layer'''
    def __init__(self, V, D, **kwargs):
        super().__init__(**kwargs)
        T = 1
        self.embedding = Embedding(V, D, input_length = T)

    def call(self, inputs):
        x = self.embedding(inputs)
        return x

class DecoderLSTM(tf.keras.layers.Layer):
    '''A simple decoder lstm layer'''
    def __init__(self, M, **kwargs):
        super().__init__(**kwargs)
        self.M = M
        self.lstm = LSTM(M, return_state = True)

    def call(self, inputs, state = None):
        if not state:
            s = tf.zeros((len(inputs),self.M))
            state = [s, s]
        return self.lstm(inputs, initial_state = state)

# class Decoder(tf.keras.layers.Layer):
#     '''Decoder all in one place'''
#     def __init__(self, V, D, M, T = 1, **kwargs):
#         super().__init__(**kwargs)
#         self.M = M
#         self.embedding = Embedding(V, D, input_length = T)
#         self.lstm = LSTM(M, return_state = True)
#         self.dense = Dense(V, activation = 'softmax')

#     def call(self, inputs, state = None):
#         if not state:
#             s = tf.zeros((len(inputs),self.M))
#             state = [s, s]
        
#         x = self.embedding(inputs)
#         o, h, c = self.lstm(x, initial_state = state)
#         output = self.dense(o)
#         return output, [h,c]

class Decoder(tf.keras.layers.Layer):
    
    def __init__(self, V, D, M, T = 1, **kwargs):
        super().__init__(**kwargs)
        self.M = M
        self.embedding = Embedding(V, D, input_length = T)
        self.lstm = LSTM(M, return_state = True)
        self.dense = Dense(V, activation = 'softmax')
        self.concat = Concatenate(axis = -1)

    def call(self, inputs, state = None):
        # context should be of shape (N, 1, M_enc)
        
        if not state:
            s = tf.zeros((len(inputs[0]),self.M))
            state = [s, s]

        # input contains 2 things
        # 1 - decoder sequence input
        # 2 - context vectors input
        # these 2 things a detached here
        decoder_inputs, context = inputs
        
        # embedding decoder sequence inputs
        x = self.embedding(decoder_inputs)

        # concat context and embedded vectors
        x = self.concat([x, context])

        # go through decoder lstm
        o, h, c = self.lstm(x, initial_state = state)

        # go through decoder dense
        output = self.dense(o)

        # return decoder final output and lstm states
        return output, [h,c]


def test_decoder_header(V = 2000, D = 200, N = 128):
    decoder_header = DecoderHeader(V, D)
    decoder_input_data = np.random.randint(0,100, (N, 1))
    assert decoder_header(decoder_input_data).shape == (N, 1, D)
    print("Decoder header(embedding) test success.")


def test_decoder_lstm(N = 128, D = 200, M = 15):
    decoder_lstm = DecoderLSTM(M)

    o, h, c = decoder_lstm(
        np.random.random((N, 1, D)), 
        state = [tf.zeros((N,M)),
        tf.zeros((N,M))]
        )

    assert  all((o == h).numpy().flatten())
    assert o.shape == h.shape == c.shape ==  (N,M)
    print("Decoder lstm test successfully")

# def test_decoder(V = 2000, M = 15,D = 200,N = 128,T = 1):
#     s = tf.zeros((N, M))
#     states = [[s,s],None]
#     for state in states:
    
#         decoder = Decoder(V, D, M ,T)
#         out, state = decoder(np.random.randint(1,V, (N,T)), state = None)
#         assert out.shape == (N, V)
#         assert state[0].shape == (N, M)
#         assert state[1].shape == (N, M)
#     print('Passed decoder test')
