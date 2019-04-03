#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:15:54 2019

@author: adrienthomas
"""

import pickle
from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent
from keras.models import Model
from numpy import array
import keras
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.callbacks import TensorBoard
import time
import json

path = '/home/charles1110/OneDrive/3eme annee ecole/OSY/Projet Illuin/2018-question-answering'

import os
os.chdir(path)


#%% Loading the preprocessed data
 
padded_questions = pickle.load(open('pickled_data/pickle_padded_questions', 'rb'))
padded_contexts = pickle.load(open('pickled_data/pickle_padded_contexts', 'rb'))
p_starts = pickle.load(open('pickled_data/pickle_padded_pstarts', 'rb'))
p_ends = pickle.load(open('pickled_data/pickle_padded_pends', 'rb'))
vocab_data = pickle.load(open('pickled_data/pickle_vocab_data', 'rb'))
embedding_matrix = pickle.load(open('pickled_data/pickle_embedding_matrix', 'rb'))

max_length_context = vocab_data['max_length_contexts']
max_length_question = vocab_data['max_length_questions']
vocab_size = vocab_data['vocab_size']
t = vocab_data['tokenizer']

#%% Setting the model parameters

RNN = recurrent.LSTM
EMBED_HIDDEN_SIZE = 50
CONT_HIDDEN_SIZE = 200
QUERY_HIDDEN_SIZE = 200
dropout_rate = 0.5
BATCH_SIZE = 60
EPOCHS = 100
validation_split = 0.05
lning_rate = 0.001

# Dictionnary to save the model params
params = {}
params['RNN'] = RNN.__name__
params['EMBED_HIDDEN_SIZE'] = EMBED_HIDDEN_SIZE
params['CONT_HIDDEN_SIZE'] = CONT_HIDDEN_SIZE
params['QUERY_HIDDEN_SIZE'] = QUERY_HIDDEN_SIZE
params['dropout_rate'] = dropout_rate
params['BATCH_SIZE'] = BATCH_SIZE
params['EPOCHS'] = EPOCHS
params['validation_split'] = validation_split
params['lning_rate'] = lning_rate
params['train_size'] = -1


#%% Define customs functions/classes used to build the model

def normal_loss(y_true, y_pred):
    print('y_true',y_true.shape)
    print('y_pred',y_pred.shape)
    y_true_index = tf.argmax(y_true,axis = 1)
    print('y_true_index',y_true_index.shape)
    loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true_index, logits=y_pred)
    #loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true[:, 1], logits=y_pred_index[:, 1])
    #loss = keras.losses.categorical_crossentropy(y_true, y_pred)
    
    return(tf.reduce_mean(loss_start))
    
    
from keras.layers import Layer
        
class NonMasking(Layer):   
    def __init__(self, **kwargs):   
        self.supports_masking = True  
        super(NonMasking, self).__init__(**kwargs)   
  
    def build(self, input_shape):   
        input_shape = input_shape   
  
    def compute_mask(self, input, input_mask=None):   
        # do not pass the mask to the next layers   
        return None   
  
    def call(self, x, mask=None):   
        return x   
  
    def get_output_shape_for(self, input_shape):   
        return input_shape  
    
#%% Load a pre-existing model
        
from keras.models import load_model

model_name = 'model_14-03-2019-12_46'

model = load_model('saved_models/' + model_name, custom_objects={'NonMasking' : NonMasking}) 

#%% Let's build the model

context = layers.Input(shape=(max_length_context,), dtype='int32')
encoded_context = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE, weights=[embedding_matrix],
                                   input_length=max_length_context, trainable=False, mask_zero = True)(context)
encoded_context = layers.Bidirectional(RNN(CONT_HIDDEN_SIZE, return_sequences=True),merge_mode='concat')(encoded_context)

encoded_context = keras.layers.Dropout(dropout_rate)(encoded_context)
encoded_context = NonMasking()(encoded_context)

question = layers.Input(shape=(max_length_question,), dtype='int32')
encoded_question = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE, weights=[embedding_matrix],
                                    input_length=max_length_question, trainable=False, mask_zero = True)(question)
encoded_question = layers.Bidirectional(RNN(QUERY_HIDDEN_SIZE, return_sequences=True),merge_mode='concat')(encoded_question)

encoded_question = keras.layers.Dropout(dropout_rate)(encoded_question)
encoded_question = NonMasking()(encoded_question)

# Implementation of the attention layer
encoded_question_t = layers.Permute((2,1), input_shape=encoded_question.shape)(encoded_question)
attention_logits = layers.dot([encoded_context, encoded_question_t], axes=[-1,1])
attention_dist = layers.Activation('softmax')(attention_logits)

attention_output = layers.dot([attention_dist, encoded_question], axes=[-1,1])
attention_output = keras.layers.Dropout(dropout_rate)(attention_output)


merged = layers.concatenate([encoded_context, attention_output], axis=2)
merged = layers.Dense(200, activation='relu')(merged)
p_start = layers.Dense(1, activation='relu')(merged)
p_start = layers.Flatten()(p_start)
p_end = layers.Dense(1, activation='relu')(merged)
p_end = layers.Flatten()(p_end)

p_start = layers.Activation('softmax')(p_start)
p_end = layers.Activation('softmax')(p_end)

model = Model([context, question], [p_start, p_end])
model.compile(optimizer=keras.optimizers.Adam(lr=lning_rate),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
model.summary()


#%% Training the model

train_size = -1
params['train_size'] = train_size

x_context = array(padded_contexts[:train_size])
x_question = array(padded_questions[:train_size])
y_pstarts = array(p_starts[:train_size])
y_pends = array(p_ends[:train_size])

time_str = time.strftime("%d-%m-%Y-%H_%M")

tensorboard = TensorBoard(log_dir="logs/{}".format(time_str))

model.fit([x_context, x_question], [y_pstarts, y_pends],batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=validation_split,
          callbacks=[tensorboard])

history = model.history


#%% Plot model training performances

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

#%% Save the model

#time_str = time.strftime("%d-%m-%Y-%H_%M")
try:
    model.save('saved_models/model_{}'.format(time_str))
    with open('saved_models/metrics_{}.json'.format(time_str), 'w') as fp:
        json.dump(history.history, fp)
    with open('saved_models/params_{}.json'.format(time_str), 'w') as fp:
        json.dump(params, fp)
except TypeError:
    try :
        # serialize model to JSON
        model_json = model.to_json()
        with open('saved_models/{}.json'.format(time_str), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights('saved_models/{}.h5'.format(time_str))
        print("Saved model to disk")
    except TypeError:
        with open('saved_models/metrics_{}.json'.format(time_str), 'w') as fp:
            json.dump(history.history, fp)
        model_config = model.get_config()
        with open('saved_models/graph_{}.json'.format(time_str), 'w') as fp:
            json.dump(model_config, fp)
        


#%% Print a few answer predictions of the model

def get_squad_answer(padded_context, p_start, p_end):
    """ Takes a given context and the respective probability vectors and returns the SQuAD answer """

    start_index = p_start.index(max(p_start))
    end_index = p_end.index(max(p_end))
    encoded_answer = padded_context[start_index:end_index+1]
    answer = []
    for encoded_word in encoded_answer:
        for word, index in t.word_index.items():
            if encoded_word == index:
                answer.append(word)
                break
    answer = " ".join(answer)

    return answer


span_test = 65000
result = model.predict([padded_contexts[span_test:span_test+10],padded_questions[span_test:span_test+10]])

for x in range(len(result[0])):
    print('True answer')
    print(p_starts[span_test+x].index(1),p_ends[span_test+x].index(1))
    print(get_squad_answer(padded_contexts[span_test+x], p_starts[span_test+x], p_ends[span_test+x]), '\n')
    print('Prediction')
    print(list(result[0][x]).index(max(result[0][x])),list(result[1][x]).index(max(result[1][x])))
    print(get_squad_answer(padded_contexts[span_test+x], list(result[0][x]), list(result[1][x])), '\n\n')
    
    
#%% Compute validation accuracy taking into account p_start AND p_end
    
validation_test = 3553
result = model.predict([padded_contexts[-validation_test: -1], padded_questions[-validation_test:-1]])
print('result computed')

good_predictions = 0
for x in range(len(result[0])):
    if (p_starts[-validation_test+x].index(1)==list(result[0][x]).index(max(result[0][x]))) and (p_ends[-validation_test+x].index(1)==list(result[1][x]).index(max(result[1][x]))):
        good_predictions += 1
        print('true')

validation_accuracy = good_predictions / len(result[0])
print(validation_accuracy)  # 0.21227477477477477
