import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

tokenizer = Tokenizer()

def GladdosPhrases():
	URL = "https://theportalwiki.com/wiki/GLaDOS_voice_lines"
	page = requests.get(URL)
	soup = BeautifulSoup(page.content, "html.parser")
	gladdos_lines = soup.find_all("i")
	phrases = []
	for line in gladdos_lines:
		phrases.append(line.text.strip().lower())
	print(phrases)		
	return phrases

def runML():

	corpus = GladdosPhrases()

	tokenizer.fit_on_texts(corpus)
	total_words = len(tokenizer.word_index) + 1

	input_sequences = []
	for line in corpus:
		token_list = tokenizer.texts_to_sequences([line])[0]
		for i in range(1, len(token_list)):
			n_gram_sequence = token_list[:i+1]
			input_sequences.append(n_gram_sequence)

	# pad sequences
	max_sequence_len = max([len(x) for x in input_sequences])
	input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

	# create predictors and label
	xs, labels = input_sequences[:,:-1],input_sequences[:,-1]

	ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

	model = Sequential()
	model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
	model.add(Bidirectional(LSTM(150)))
	model.add(Dense(total_words, activation='softmax'))
	adam = Adam(learning_rate=0.01)
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	history = model.fit(xs, ys, epochs=100, verbose=1)
	#print model.summary()
	print(model)

	def plot_graphs(history, string):
		plt.plot(history.history[string])
		plt.xlabel("Epochs")
		plt.ylabel(string)
		plt.show()

	plot_graphs(history, 'accuracy')

	model.save("GLADDOS")
	#Generate Text
	seed_text = "I've got a bad feeling about this"
	next_words = 100

	for _ in range(next_words):
		token_list = tokenizer.texts_to_sequences([seed_text])[0]
		token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
		y_prob = model.predict(token_list)
		predicted = y_prob.argmax(axis=-1)
		output_word = ""
		for word, index in tokenizer.word_index.items():
			if index == predicted:
				output_word = word
				break
		seed_text += " " + output_word
	print(seed_text)

runML()