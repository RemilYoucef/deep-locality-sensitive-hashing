import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM, Dense, Lambda, Conv1D, GlobalMaxPooling1D, concatenate
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
import scipy.stats as stats 
import itertools

from similarities import *

class HamDist(Layer):

    def __init__(self, b, m):
        self.b = b
        self.m = m
        self.result = None
        super(HamDist, self).__init__()

    def build(self, input_shape):
        super(HamDist, self).build(input_shape)

    def call(self, x):
        i = 0
        count = 0
        slicing = self.b
        size_embedding = self.b * self.m
        while i < size_embedding :
            count += K.max(K.abs(x[0][:,i:i+slicing] - x[1][:,i:i+slicing]), axis = 1) * slicing
            i = i + slicing
        self.result = 1 - count / (size_embedding * 2)
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)
    
class ManhDist(Layer):

    def __init__(self, **kwargs):
        self.result = None
        super(ManhDist, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ManhDist, self).build(input_shape)

    def call(self, x, **kwargs):
        
        self.result = K.exp(-0.005*K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)
    
class ProdVec(Layer):

    def __init__(self, size_embedding):
        self.size_embedding = size_embedding
        self.result = None
        super(ProdVec, self).__init__()

    def build(self, input_shape):
        super(ProdVec, self).build(input_shape)

    def call(self, x):
        
        self.result = K.mean(K.sum((x * x), axis=1, keepdims=True) / self.size_embedding)
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)
    
class SumVec(Layer):

    def __init__(self, size_embedding):
        self.size_embedding = size_embedding
        self.result = None
        super(SumVec, self).__init__()

    def build(self, input_shape):
        super(SumVec, self).build(input_shape)

    def call(self, x):
        
        self.result = K.sum(x, axis = 1) / self.size_embedding
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)
    
class AbsVect(Layer):

    def __init__(self, **kwargs):
        self.result = None
        super(AbsVect, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AbsVect, self).build(input_shape)

    def call(self, x, **kwargs):
        
        self.result = K.abs(x)
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)
    
class CosDist(Layer):

    def __init__(self, **kwargs):
        self.result = None
        super(CosDist, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CosDist, self).build(input_shape)

    def call(self, x, **kwargs):
        
        initializer = tf.keras.initializers.Ones()
        values_1 = initializer(shape=(128,1))
        self.result = K.dot(K.l2_normalize(K.abs(x), axis=-1), K.l2_normalize(values_1, axis=-1))
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)
    
def custom_loss(y_true, y_pred):
    
    cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    return cosine_loss(y_true, y_pred)


def siamese_model(shared_model, input_shape, b, m, is_sparse = False, print_summary = True):
    size_hash_vector = m * b
    stack_1_input = Input(sparse = is_sparse, shape = input_shape)
    stack_2_input = Input(sparse = is_sparse, shape = input_shape)
    ham_distance = HamDist(b,m)([shared_model(stack_1_input), shared_model(stack_2_input)])
    model = Model(inputs = [stack_1_input, stack_2_input], outputs = [ham_distance,
                                                                      ProdVec(size_hash_vector)(shared_model(stack_1_input)), 
                                                                      ProdVec(size_hash_vector)(shared_model(stack_2_input)),
                                                                      SumVec(size_hash_vector)(shared_model(stack_1_input)),
                                                                      SumVec(size_hash_vector)(shared_model(stack_2_input))])

    metrics = [tf.keras.metrics.RootMeanSquaredError(name="rmse"), tf.keras.metrics.MeanAbsoluteError(name="mae")]
    model.compile(loss = ["mse", "mse", "mse","mse","mse"],
                  loss_weights=[15/16, 1/64, 1/64, 1/64, 1/64],
                  optimizer=tf.keras.optimizers.Adam(), 
                  metrics = [metrics, None, None, None, None])
    
    if print_summary : 
        print(model.summary())
        print(shared_model.summary())
    
    return model


def siamese_model_baseline(shared_model, input_shape, is_sparse = False, print_summary = True):

    stack_1_input = Input(sparse = is_sparse, shape = input_shape)
    stack_2_input = Input(sparse = is_sparse, shape = input_shape)
    manh_distance = ManhDist()([shared_model(stack_1_input), shared_model(stack_2_input)])
    model = Model(inputs = [stack_1_input, stack_2_input], outputs = [manh_distance,
                                                                      AbsVect()(shared_model(stack_1_input)), 
                                                                      AbsVect()(shared_model(stack_2_input))])
    metrics = [tf.keras.metrics.RootMeanSquaredError(name="rmse"), tf.keras.metrics.MeanAbsoluteError(name="mae")]
    model.compile(loss = ['mse', custom_loss, custom_loss],
                  loss_weights=[1/2, 1/4, 1/4],
                  optimizer=tf.keras.optimizers.Adam(), 
                  metrics = [metrics, None, None])
    
    if print_summary : 
        print(model.summary())
        print(shared_model.summary())
    
    return model
        

def train_siamese_model(model, X_train, X_validation, Y_train, Y_validation, batch_size, epochs):
    _1_train = np.ones((Y_train.size,))
    _1_validation = np.ones((Y_validation.size,))
    _0_train = np.zeros((Y_train.size,))
    _0_validation = np.zeros((Y_validation.size,))
    siamese_model = model.fit([X_train['stack_1'], X_train['stack_2']], [Y_train, _1_train, _1_train, _0_train, _0_train],
                      batch_size = batch_size,
                      epochs = epochs,
                      validation_data=([X_validation['stack_1'], X_validation['stack_2']], [Y_validation, _1_validation, _1_validation, _0_validation, _0_validation]))
    return siamese_model

def train_siamese_model_baseline(model, X_train, X_validation, Y_train, Y_validation, size_hash_vector, batch_size, epochs):
    _1_train = np.ones((Y_train.size, size_hash_vector))
    _1_validation = np.ones((Y_validation.size, size_hash_vector))
    siamese_model = model.fit([X_train['stack_1'], X_train['stack_2']], [Y_train, _1_train, _1_train],
                      batch_size = batch_size,
                      epochs = epochs,
                      validation_data=([X_validation['stack_1'], X_validation['stack_2']], [Y_validation, _1_validation, _1_validation]))
    return siamese_model


def predict(model, X):
    return model.predict([X['stack_1'], X['stack_2']])[0].reshape(1,-1)


def spearman_rho(predictions, real_values):
    rho, p_value = stats.spearmanr(predictions[0], real_values)
    return (rho, p_value)

def kendall_tau(predictions, real_values):
    tau, p_value = stats.kendalltau(predictions[0], real_values)
    return (tau, p_value)


def transform (x):
    return 1 if x > 0 else -1

def hamming(embedding1, embedding2, slicing, length) :
    count = 0
    i = 0
    while i < length :
        if np.unique(embedding1[i:i+slicing] == embedding2[i:i+slicing]).shape[0] == 1 & np.unique(embedding1[i:i+slicing] == embedding2[i:i+slicing])[0] == True :
            count += 1
        i += slicing
    return count / length * slicing

def hamming_diff(embedding1, embedding2, slicing, length) :
    i = 0
    count = 0
    while i < length :

        count += np.max(np.abs(embedding1[i:i+slicing] - embedding2[i:i+slicing])) * slicing
        i += slicing
    return 1 - count / (length * 2)

        
def intermediate_model_trained(shared_model, output_layer, CNN = False, input_tensor = None):
    if CNN :
        return Model(inputs = input_tensor, outputs = shared_model.layers[output_layer].output)
    else :
        return Model(inputs = shared_model.input, outputs = shared_model.layers[output_layer].output)  
    

def compare_hamming(X, intermediate_model, b, size_embedding):
    
    df_hamming = pd.DataFrame()
    df_hamming['embedding_stack_1'] = pd.Series(intermediate_model.predict(X['stack_1']).tolist())
    df_hamming['embedding_stack_2'] = pd.Series(intermediate_model.predict(X['stack_2']).tolist())
    df_hamming['embedding_stack_1'] = df_hamming['embedding_stack_1'].apply(lambda x : np.array(list(map(transform, x))))
    df_hamming['embedding_stack_2'] = df_hamming['embedding_stack_2'].apply(lambda x : np.array(list(map(transform, x))))
    df_hamming['hamming'] = df_hamming.apply(lambda x : hamming(x['embedding_stack_1'], x['embedding_stack_2'], b, size_embedding), axis = 1)
    return df_hamming


def index_frame(l, df) :
    list_index = []
    for elt in l :
        try :
            list_index.append(df.index[df['frame'] == elt][0] + 1)
        except :
            list_index.append(0)
    return list_index

def assign_stacks (index, df) :
    n = df.shape[0]
    a,b = get_indices_sim(n, index)
    return (df['rankFrames'][a], df['rankFrames'][b])

def padding(df, max_seq_length):
    
    dict_X = {'stack_1': df['stack1'], 'stack_2': df['stack2']}

    for data, side in itertools.product([dict_X], ['stack_1', 'stack_2']):
        data[side] = pad_sequences(data[side], padding = 'post', truncating = 'post', maxlen = max_seq_length)

    return data