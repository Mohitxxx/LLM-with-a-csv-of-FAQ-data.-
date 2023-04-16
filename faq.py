#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

# Import the FAQ data from CSV
df = pd.read_csv('faq_data.csv')

# Preprocess the data
df['question'] = df['question'].str.lower().replace('[^a-zA-Z0-9\s]', '', regex=True)
df['answer'] = df['answer'].str.lower().replace('[^a-zA-Z0-9\s]', '', regex=True)

# Split the data into questions and answers
questions = df['question'].values
answers = df['answer'].values

# Split the data into training and testing sets
questions_train, questions_test, answers_train, answers_test = train_test_split(questions, answers, test_size=0.2)

# Convert the text data into numerical form using tokenization
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(questions_train)
questions_train_seq = tokenizer.texts_to_sequences(questions_train)
questions_test_seq = tokenizer.texts_to_sequences(questions_test)

# Pad the sequences to make them the same length
maxlen = 50
questions_train_seq = pad_sequences(questions_train_seq, padding='post', maxlen=maxlen)
questions_test_seq = pad_sequences(questions_test_seq, padding='post', maxlen=maxlen)

# Define the LLM model architecture
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64, input_length=maxlen))
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(questions_train_seq, answers_train, epochs=50, batch_size=32, validation_data=(questions_test_seq, answers_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(questions_test_seq, answers_test)
print('Test accuracy:', test_acc)

# Use the model to generate responses to new FAQ questions
new_question = ["How do I reset my password?"]
new_question_seq = tokenizer.texts_to_sequences(new_question)
new_question_seq = pad_sequences(new_question_seq, padding='post', maxlen=maxlen)
prediction = model.predict(new_question_seq)
print('Answer:', prediction)

