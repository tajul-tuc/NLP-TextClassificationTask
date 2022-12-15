#Handle dataset
import pandas as pd
import numpy as np
import csv
import re
#Stopwords
from nltk.corpus import stopwords
#Data Augmentation
from textaugment import EDA
#Split into train and test
from sklearn.model_selection import train_test_split
#Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
#Encode words(text) into integers
from tensorflow.keras.preprocessing.text import text_to_word_sequence
#Pad sequence, turn list of integer into 2D Numpy array
from tensorflow.keras.preprocessing.sequence import pad_sequences

#LSTM Model
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2
import os
import pickle





#import the train.csv dataset.

#import train dataset
data_train = pd.read_csv("train.csv")





#settings

#for each tweet, only 40 words can be passed to the model
max_sequence_length = 40
#limit dictionary of words to 3000
max_words = 4000
#define the size of output vector from this layer for each word
embedding_size = 40
#Save your trained model to this file
model_file = 'model.h5'
#Store the word dictionary
tokenizer_file = 'tokenizer.pickle'
#binary classification
num_classes = 2





#Text cleaning, clean irrelevant signals, and links

def clean_str(string):
    #replace and simplify a lot of signals
    string = re.sub(r'http\S+', 'link', string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    
cleanr = re.compile('<.*?>')

string = re.sub(r'\d+', '', string)
    string = re.sub(cleanr, '', string)
    string = re.sub("'", '', string)
    string = re.sub(r'\W+', ' ', string)
    string = string.replace('_', '')
    #turn to lower case
    return string.strip().lower()




    
#Remove stopwords and Data augmentation.

stop_words = set(stopwords.words('english'))  #stopwords for English
def remove_stopwords(word_list):
    no_stop_words = [w for w in word_list if not w in stop_words]
    return no_stop_words
    
#EDA
t = EDA()
t.random_deletion("...")
t.random_swap("...")
t.synonym_replacement("...")





#The next thing we need to do is to split train.csv into a 
#training dataset, validation dataset, and test dataset.

#Split into train & valiation and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25,random_state=42)
#Split previous train into train and validation
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.25,random_state=42)
X_train = [item for sublist in X_train for item in sublist]
Y_train = [item for sublist in Y_train for item in sublist]
X_validation = [item for sublist in X_validation for item in sublist]
Y_validation = [item for sublist in Y_validation for item in sublist]
X_test = [item for sublist in X_test for item in sublist]
Y_test = [item for sublist in Y_test for item in sublist]





#Apply tokenizer on our training dataset

# Tokenizer
tokenizer = Tokenizer(num_words=max_words)
# Updates internal vocabulary based on a list of texts. 
tokenizer.fit_on_texts(X_train)
# Transforms each row from texts to a sequence of integers. 
X_train = tokenizer.texts_to_sequences(X_train)
X_validation = tokenizer.texts_to_sequences(X_validation)
X_test = tokenizer.texts_to_sequences(X_test)
# Pad sequences
X_train = pad_sequences(X_train, maxlen=max_sequence_length, dtype='int32', value=0)
X_validation = pad_sequences(X_validation, maxlen=max_sequence_length, dtype='int32', value=0)
X_test = pad_sequences(X_test, maxlen=max_sequence_length, dtype='int32', value=0)
word_index = tokenizer.word_index
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_validation = np.array(X_validation)
Y_validation = np.array(Y_validation)
X_test = np.array(X_test)
Y_test = np.array(Y_test)





#LSTM Model

l2_reg = l2(0.001)
def model_fn():
    model = Sequential()
#embeddings_regularizer: Regularizer function applied to the embeddings matrix (see keras.regularizers).
    model.add(Embedding(max_words, embedding_size, input_length=max_sequence_length, embeddings_regularizer=l2_reg))
    
    model.add(SpatialDropout1D(0.5))
    
    model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg, bias_regularizer=l2_reg))
    
    model.add(Dropout(0.2))
    
    model.add(Dense(1024, activation='relu'))
    
    model.add(Dropout(0.2))
    
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
    
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
print(model.summary())
return model





#Train the model

# epochs
epochs = 10
# number of samples to use for each gradient update
batch_size = 128
# saving tokenizer
with open(tokenizer_file, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
model = model_fn()
# loadin saved model
if os.path.exists(model_file):
    model.load_weights(model_file)
history = model.fit(X_train, Y_train,
          validation_data=(X_validation, Y_validation),
          epochs=epochs,
          batch_size=batch_size,
          shuffle=True,
          verbose=1)
# saving model
model.save_weights(model_file)





#Test the model

#Test model
scores = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print("Acc: %.2f%%" % (scores[1] * 100))