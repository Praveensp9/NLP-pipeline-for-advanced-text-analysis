#!/usr/bin/python3
#Author : Praveen Patil

import string
import tensorflow as flow
import keras as ke
import keras.utils as ku 
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
import numpy as np
from numpy import array

def load_training_file(file):
	f = open(file,'r')
	train = f.read()
	f.close()
	return train

def clean_train_file(file):
	file = file.replace('--', ' ')
	tokens = file.split()
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	tokens = [word for word in tokens if word.isalpha()]
	tokens = [word.lower() for word in tokens]
	return tokens

def save_training_file(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

filename='train.text'
train = load_training_file(filename)
tokens = clean_train_file(train)
print('Total Tokens: %d' % len(tokens)) 
print('Unique Tokens: %d' % len(set(tokens))) 
length = 50 + 1
sequences = list()
for i in range(length, len(tokens)):
	seq = tokens[i-length:i]
	line = ' '.join(seq)
	sequences.append(line)

outfile = 'Processed_train.text'
save_training_file(sequences, outfile)

r = load_training_file("Processed_train.text")
lines = r.split('\n')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
size = len(tokenizer.word_index) + 1
sequences=np.array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=size)
seq = X.shape[1]


model = Sequential()
model.add(Embedding(size, 50,input_length=seq))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(size, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, batch_size=128, epochs=100)
model.save('training_model.h5')
dump(tokenizer, open('tokenizer.pkl', 'wb'))
print("********The End********") 
