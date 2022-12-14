import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Pandas DataFrame to see 5 rows
tweet = pd.read_csv('C:/Users/T.Islam/Documents/TweetDataSet/train.csv')
test = pd.read_csv('C:/Users/T.Islam/Documents/TweetDataSet/test.csv')
#print(tweet.head(3))

print('There are {} rows and {} columns in train'.format(tweet.shape[0],tweet.shape[1]))
print('There are {} rows and {} columns in test'.format(test.shape[0],test.shape[1]))

#Test representation: countvectorizer
def cleaningText(tweet):
    tweet['text'] = [re.sub(r'http\S+', '', x, flags=re.MULTILINE) for x in tweet['text']]
    tweet['text'] = tweet['text'].str.lower()
    
cleaningText(tweet)
tweet.head()
#print(tweet.head())

sentences = [x for x in tweet['text']]
labels = [x for x in tweet['target']]
#print(sentences[:100])

#training and testing data based on 80/20 rule
labels = np.array(labels)
training_sentences = sentences[:6090]
training_labels = labels[:6090]
testing_sentences = sentences[6090:]
testing_labels = labels[6090:]

#Our nlp model parameters
vocab_size = 10000 #max num of words we can store in our own dictionary
embedding_dim = 16 #low dimentional space
max_length = 280 #tweet length
trunc_type='post' #padding on the end
oov_tok = "<OOV>" # for new word = out of voc
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)

#model 
model = tf.keras.Sequential([
tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length = max_length),
tf.keras.layers.GlobalAveragePooling1D(),
tf.keras.layers.Dense(6, activation = "relu"),
tf.keras.layers.Dense(1, activation = "sigmoid")
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#model.summary()

#We will run our model 10 times
np.random.seed(42)
num_epochs = 5
model.fit( padded, training_labels,epochs = num_epochs, validation_data = (testing_padded, testing_labels))