#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###############################################################################################################################
#                               IMPORT ALL THE REQUIRED LIBRARIES
###############################################################################################################################

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import string
from string import digits
#import matplotlib.pyplot as plt
#%matplotlib inline
import re

import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
from keras.models import load_model

import flask
from flask import Flask, request, jsonify

import pickle

print(os.listdir("C:/Users/bvsba/Desktop/UIC/Summer Semester/Language translation/"))

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', -1)

# Any results you write to the current directory are saved as output.

###############################################################################################################################
#                                           GENERATE BATCH STAR
###############################################################################################################################

def generate_batch_test(X, batch_size = 1):
    ''' Generate a batch of data '''
    while True:
        for j in range(0, len(X), batch_size):
            max_length_src = 20
            max_length_tar = 20
            num_encoder_tokens = 35121
            num_decoder_tokens = 53091
            
            encoder_input_data = np.zeros((batch_size, max_length_src),dtype='float32')
            decoder_input_data = np.zeros((batch_size, max_length_tar),dtype='float32')
            decoder_target_data = np.zeros((batch_size, max_length_tar, num_decoder_tokens),dtype='float32')
            for i, (input_text) in enumerate(zip(X[j:j+batch_size])):
                input_text = input_text[0]
                for t, word in enumerate(input_text.split()):
                    encoder_input_data[i, t] = input_token_index[word] # encoder input seq
                #for t, word in enumerate(target_text.split()):
                #    if t<len(target_text.split())-1:
                #        decoder_input_data[i, t] = target_token_index[word] # decoder input seq
                    #if t>0:
                        # decoder target sequence (one hot encoded).
                        
                        # does not include the START_ token
                        # Offset by one timestep
                        #decoder_target_data[i, t - 1, target_token_index[word]] = 1.
            yield([encoder_input_data, decoder_input_data])

###############################################################################################################################
#                                           ORIGINAL MODEL
###############################################################################################################################            


# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
#model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


app = Flask(__name__)

def load_model1(weights_path):
   
   global model1, encoder_model, decoder_model, input_token_index, target_token_index, reverse_target_char_index
   weights_path = "C:/Users/bvsba/langtrans_weights.h5"
   #model1 = Model([encoder_inputs, decoder_inputs], decoder_outputs)
   model1 = load_model('C:/Users/bvsba/my_model')
   encoder_model = load_model('C:/Users/bvsba/encoder_model')
   decoder_model = load_model('C:/Users/bvsba/decoder_model')
   a_file = open("C:/Users/bvsba/input_token_index.pkl", "rb")
   input_token_index = pickle.load(a_file) 
   a_file.close()
   b_file = open("C:/Users/bvsba/target_token_index.pkl", "rb")
   target_token_index = pickle.load(b_file) 
   b_file.close() 
   c_file = open("C:/Users/bvsba/reverse_target_char_index.pkl", "rb")
   reverse_target_char_index = pickle.load(c_file) 
   c_file.close() 
    
@app.route('/')
def home_endpoint():
    return 'Hello World!'

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['START_']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '_END' or
           len(decoded_sentence) > 50):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence

@app.route('/predict', methods=['POST'])
def get_prediction1():
    # Works only for a single sample
    if request.method == 'POST':
        print("entered") 
        text = request.json
        text = text['eng']
        text = text.lower()
        X_test = pd.Series(text)
        test_gen = generate_batch_test(X_test, batch_size = 1)
        (input_test), _ = next(test_gen)
        decoded_sentence = decode_sequence(input_test)
        prediction = decoded_sentence[:-4]
        return prediction 
       
 
if __name__ == '__main__':
   
    weights_path = "C:/Users/bvsba/langtrans_weights.h5"
    load_model1(weights_path)  # load model at the beginning once only    
    app.run(host='127.0.0.1', port=5000)    

