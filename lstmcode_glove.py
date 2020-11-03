#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from datetime import datetime
__author__ = 'maxim'

import numpy as np
import string

from keras.callbacks import LambdaCallback
#from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import LSTM,Dense, Activation, Bidirectional
from keras.models import Sequential
from keras.utils.data_utils import get_file

print('\nFetching the text...', datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)"))
#plist = open('patientvectors.txt','r').read().split('\n')
plist = open('icdbiosent2vec.txt','r').read().split('\n')
plist.remove('')
icds = [[p[0:len(p)] for p in patient.split(', ')] for patient in plist]
pretrained_weights = np.array([line[1:len(line)] for line in icds])
#vocab_size=6072
vocab_size=4876
embedding_size=700

print('\nPreparing the sentences...', datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)"))
# with open('patients.txt') as file_:
  # docs = file_.readlines()
# sentences = [[word for word in doc.split()[:max_sentence_len]] for doc in docs]
# print('Num sentences:', len(sentences))

#with open('inputsentences.txt') as f2:
with open('inputicdsentences.txt') as f2:
	test2 = list(f2)
sentences = [[word  for word in row.split()] for row in test2 if row and not row.isspace()]
print('Num sentences:', len(sentences))

def word2idx(word):
  for i in range(0,len(icds)):
	  if word == icds[i][0]:
		  return i
def idx2word(idx):
  return icds[idx][0]
  
print('\nPreparing the data for LSTM...', datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)"))
#max_sentence_len=579
max_sentence_len=496
train_x = np.zeros([len(sentences), max_sentence_len], dtype=np.int32)
train_y = np.zeros([len(sentences)], dtype=np.int32)

for i, sentence in enumerate(sentences):
  for t, word in enumerate(sentence[:-1]):
    train_x[i, t] = word2idx(word)
  train_y[i] = word2idx(sentence[-1])
print('train_x shape:', train_x.shape)
print('train_y shape:', train_y.shape)
print(train_x[0])
print(train_y[0])


print('\nTraining LSTM...', datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)"))
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[pretrained_weights]))
model.add(LSTM(units=embedding_size))
model.add(Dense(units=vocab_size))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

def sample(preds, temperature=1.0):
  if temperature <= 0:
    return np.argmax(preds)
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

def generate_next(text, num_generated=10):
  word_idxs = [word2idx(word) for word in text.split()]
  op=[]
  for i in range(num_generated):
    prediction = model.predict(x=np.array(word_idxs))
    idx = sample(prediction[-1], temperature=0.8)
    word_idxs.append(idx)
    op.append(idx)
  return ' '.join(idx2word(idx) for idx in op)

def on_epoch_end(epoch, _):
  print('\nGenerating text after epoch: %d' % epoch)
  texts = [
    '41401 4111 4160 25060 3572 25050 36201 25070 44381 3612',
    '85246 8054 E8150',
    '43380 4254 9972 1120 99812 42830 4423 45829 7885 2859 V4582 42731 41072 25000 4280 40390 5859 28529 496 2449 7245 V707 43310 3952',
    '45829 4532 2761 5723 4561 45621 5849 7455 E9394 5712 30393 V1582 5859'
  ]
  for text in texts:
    sample = generate_next(text)
    print('%s... -> %s' % (text, sample))

print(model.summary())
model.fit(train_x, train_y,
          batch_size=200,
          epochs=20,
          callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])
