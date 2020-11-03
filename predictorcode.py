import numpy as np, tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

data = ""
with open("patients.txt") as f:
    data = f.read().strip()
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# create line-based sequences
sequences = list()
for encoded in tokenizer.texts_to_sequences(data.split("\n")):
    if len(encoded) > 0:
        print("\n",encoded)
        for i in range(0, len(encoded) - 2):
            sequences.append(encoded[i:i+50])
print('Total Sequences: %d' % len(sequences))
sequences = np.array(sequences)
X, y = sequences[:,:-1], to_categorical(sequences[:,-1], num_classes=vocab_size)
# define model
i = tf.keras.layers.Input(shape=(X.shape[1]))
e = tf.keras.layers.Embedding(vocab_size, 10, input_length=max_length(i))
l = tf.keras.layers.LSTM(10)(e)
d = tf.keras.layers.Dense(vocab_size, activation='softmax')(l)
model = tf.keras.Model(inputs=i, outputs=[d])
print(model.summary())
# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

tf.keras.utils.plot_model(model)
model.fit(X, y, epochs=1000, verbose=2)
text = " there's still so"
text = " ".join(text.split(" ")[:3])
encoded = tokenizer.texts_to_sequences([text])[0]
encoded = array([encoded])
next = model.predict(encoded, verbose=0)
for x in next:
    next_word_token = np.argmax(x)
    # map predicted word index to word
    for word, index in tokenizer.word_index.items():
        if index == next_word_token:
            print(word + " ")
