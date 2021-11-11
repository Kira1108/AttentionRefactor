import tensorflow as tf
from encoder import Encoder
from decoder import DecoderHeader, Decoder
from attention import AttentionLayer


class TrainModel(tf.keras.models.Model):
    
    def __init__(self, T, D, V, M, Ty = 50, **kwargs):
        super().__init__(**kwargs)
        self.M = M
        self.Ty = Ty
        self.encoder = Encoder(T,V,D,M)
        self.attention = AttentionLayer(T, M)
        self.decoder_header = DecoderHeader(V, D, Ty)
        self.decoder = Decoder(V, M)
        
    def call(self, encoder_input, decoder_input, state = None):
        
        if not state:
            zero = tf.zeros((len(encoder_input),self.M))
            state = [zero, zero]
            
        s, c = state
        # encode encoder inputs
        encoder_output_tensor = self.encoder(encoder_input)
        
        # embeding target inputs
        embed_decoder_input = self.decoder_header(decoder_input)
        
        # for eatch step, attetion and decode
        out_tensors = []
        for t in range(self.Ty):
            context = self.attention(encoder_output_tensor, s)
            out, state = self.decoder(embed_decoder_input[:, t:t+1,:], context, state)
            out_tensors.append(out)
        
        # collect different time step and fix dimensions
        out_tensors = tf.stack(out_tensors)
        return tf.transpose(out_tensors, perm = [1,0,2])
    
    
if __name__ == "__main__":
    
    N = 128
    T = 100
    D = 200
    V = 2000
    M = 15
    Ty = 50
    
    final_model = TrainModel(T, D, V, M, Ty)
    encoder_input_tensor = tf.random.uniform((N, T), maxval = V, dtype = tf.int32)
    decoder_input_tensor = tf.random.uniform((N, Ty), maxval = V, dtype = tf.int32)
    output = final_model(encoder_input_tensor, decoder_input_tensor)
    print(output.shape)