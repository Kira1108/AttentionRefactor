import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, 
    Softmax,
    Concatenate,
    RepeatVector,
    Dot
)


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, T, M, **kwargs):
        super().__init__(**kwargs)
        self.M = M
        self.repeator = RepeatVector(T)
        self.concater = Concatenate()
        self.densor1 = Dense(M)
        self.densor2 = Dense(1)
        self.dotter = Dot(axes = 1)
        self.softmax = Softmax(axis = 1)
        
    def call(self, encoder_output, prev_decoder_state):
        # repeat decoder previous states T_encoder times
        s_repeated = self.repeator(prev_decoder_state)
        
        # concat decoder states with encoder hidden states
        concat_vector = self.concater([encoder_output, s_repeated])
        
        # a dense layer (for each concated vector)
        weights = self.densor1(concat_vector)
        
        # compute attention weights
        weights = self.densor2(weights)
        
        # softmax over time
        weights = self.softmax(weights)
        
        # compute context vector
        context = self.dotter([weights, encoder_output])
        return context
    
def test_attention(T = 100, M = 15, N = 128):

    prev_state = tf.zeros((N, M))
    encoder_out_tensor = tf.random.normal((N, T, 2*M))
    context = AttentionLayer(T, M)(encoder_out_tensor,prev_state)
    assert context.shape == (N, 1, 2*M)
    print("Attention test success")