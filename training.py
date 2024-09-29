import random
import json 
import pickle
import numpy as np
import nltk

nltk.download('punkt_tab')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())
words = []
classes = []
documents = []
ignore_letters = ['?','!','.',',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))


pickle.dump(words, open('words.pk1', 'wb'))
pickle.dump(classes, open('classes.pk1', 'wb'))