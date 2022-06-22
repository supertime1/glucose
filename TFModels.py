import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, Bidirectional, LSTM, Conv1D, BatchNormalization, Activation, Add, ZeroPadding1D, MaxPooling1D, AveragePooling1D, Flatten, RepeatVector, TimeDistributed
from tensorflow.keras import Input, Model
import tensorflow_lattice as tfl
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.initializers import glorot_uniform
from typing import Tuple


class RestNet1D(Model):
    def __init__(self, input_shape: Tuple, filter_size_base=16, include_top_layer=True) -> None:
        super(RestNet1D, self).__init__()
        self.filter_size_base = filter_size_base
        self.input_shape = input_shape
        self.resnet18 = self.resnet_model()
        self.include_top_layer = include_top_layer
    
    def convolutional_block(self, X, f, filters, stage, block, s = 2):  
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        # Retrieve Filters
        F1, F2 = filters
        
        # Save the input value
        X_shortcut = X

        ##### MAIN PATH #####
        # First component of main path 
        X = Conv1D(filters = F1, kernel_size = f, strides = s, padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 2, name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path (≈3 lines)
        X = Conv1D(filters = F2, kernel_size = f, strides = 1, padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 2, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        ##### SHORTCUT PATH #### (≈2 lines)
        X_shortcut = Conv1D(filters = F1, kernel_size = f, strides = s, padding = 'valid', name = conv_name_base + '1',
                            kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis = 2, name = bn_name_base + '1')(X_shortcut)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)
        
        return X

    def identity_block(self, X, f, filters, stage, block):
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        # Retrieve Filters
        F1, F2 = filters
        
        # Save the input value. You'll need this later to add back to the main path. 
        X_shortcut = X
        
        # First component of main path
        X = Conv1D(filters = F1, kernel_size = f, strides = 1, padding = 'same', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 2, name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path 
        X = Conv1D(filters = F2, kernel_size = f, strides = 1, padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 2, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation 
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)
        
        return X
    
    def resnet_model(self):
        # Define the input as a tensor with shape input_shape
        X_input = Input(self.input_shape)
        # Zero-Padding
        X = ZeroPadding1D(3)(X_input)

        # Stage 1
        X = Conv1D(self.filter_size_base * 4, 7, strides=2, name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=2, name='bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling1D(3, strides=2)(X)

        # Stage 2
        X = self.identity_block(X, 3, [self.filter_size_base * 4, self.filter_size_base * 4], stage=2, block='a')
        X = self.identity_block(X, 3, [self.filter_size_base * 4, self.filter_size_base * 4], stage=2, block='b')

        # Stage 3 (2 lines)
        X = self.convolutional_block(X, f = 3, filters = [self.filter_size_base * 8, self.filter_size_base * 8], stage = 3, block='a', s = 2)
        X = self.identity_block(X, 3, [self.filter_size_base * 8, self.filter_size_base * 8], stage=3, block='b')

        # Stage 4 (2 lines)
        X = self.convolutional_block(X, f = 3, filters = [self.filter_size_base * 16, self.filter_size_base * 16], stage = 4, block='a', s = 2)
        X = self.identity_block(X, 3, [self.filter_size_base * 16, self.filter_size_base * 16], stage=4, block='b')

        # # Stage 5 (2 lines)
        # X = convolutional_block_18(X, f = 3, filters = [filter_size_base * 32, filter_size_base * 32], stage = 5, block='a', s = 2)
        # X = identity_block_18(X, 3, [filter_size_base * 32, filter_size_base * 32], stage=5, block='b')

        # AVGPOOL (1 line).
        X = AveragePooling1D(2, name="avg_pool")(X)

        # output layer
        X = Flatten()(X)
        
        if self.include_top_layer:
            X = Dense(self.input_shape[0], activation='relu', name='fc' + str(self.input_shape[0]), kernel_initializer = glorot_uniform(seed=0))(X)
        #Create model
        model = Model(inputs = X_input, outputs = X, name='ResNet18')

        return model
    
    def call(self, x):
        return self.resnet_model(x)
    

class RNNEncoder(Model):
    def __init__(self, input_shape: Tuple) -> None:
       super(RNNEncoder, self).__init__()
       self.input_data_shape = input_shape
       self.encoder = self.rnn_encoder_model() 

    def rnn_encoder_model(self):
        X_input = Input(self.input_data_shape)
        X = Bidirectional(LSTM(32, return_sequences=True))(X_input)
        X = Bidirectional(LSTM(32, return_sequences=True))(X)
        X = Dense(1, activation='relu')(X)
        model = Model(inputs=X_input, outputs=X, name='RNNEncoder')
        return model
    
    def call(self, x):
        return self.encoder(x)
    
class RNNEncoderDecoder(Model):
  def __init__(self, latent_dim, num_of_labels):
    super(RNNEncoderDecoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = RNNEncoder(self.latent_dim)
    self.decoder = tf.keras.Sequential([
        Dense(num_of_labels, activation='relu'),
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


class CNNEncoderDecoder(Model):
  def __init__(self, latent_dim, num_of_labels):
    super(CNNEncoderDecoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = RestNet1D(input_shape=latent_dim, filter_size_base=16, include_top_layer=False)
    self.decoder = Dense(num_of_labels, activation='relu', kernel_initializer = glorot_uniform(seed=0))


  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


class AutoEncoderTSML(Model):
    def __init__(self, timesteps, n_features, latent_dim, dropout_rate=0.5):
        super(AutoEncoderTSML, self).__init__()
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.n_features = n_features
        self.dropout_rate = dropout_rate
        self.encoder = self._rnn_encoder()
        self.decoder = self._rnn_decoder()

    def _rnn_encoder(self):
        Input_X = Input((self.timesteps, self.n_features))
        X = Dropout(self.dropout_rate)
        X = LSTM(self.latent_dim * 16, return_sequences=True)(Input_X)
        X = LSTM(self.latent_dim * 4, return_sequences=True)(X)
        X = LSTM(self.latent_dim, return_sequences=True)(X)
        # X = RepeatVector(self.timesteps)(X)
        model = Model(Input_X, X, name='lstm_encoder')
        return model

    def _rnn_decoder(self):
        Input_X = Input((self.timesteps, self.latent_dim))
        X = LSTM(self.latent_dim, return_sequences=True)(Input_X)
        X = LSTM(self.latent_dim * 4, return_sequences=True)(X)
        X = LSTM(self.latent_dim * 16, return_sequences=True)(X)
        X = TimeDistributed(Dense(self.n_features, activation='selu'))(X)
        model = Model(Input_X, X, name='lstm_decoder')
        return model
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded