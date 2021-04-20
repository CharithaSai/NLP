import torch
import spacy
import sys
import time
import re
import nltk
from nltk.tokenize import word_tokenize
from spacy.lang.en import English
from keras.preprocessing.text import text_to_word_sequence
from gensim.utils import tokenize
from transformers import BertTokenizer, BertModel, BertForMaskedLM

file = open("Gita.txt")
data = file.read()


#------------WordTokenisation using  re,findall()---------

# Regex: [A-Za-z]+|[^A-Za-z ]
#
# In [^A-Za-z ] add chars you don't want to match.
#
# Details:
#
# [] Match a single character present in the list
# [^] Match a single character NOT present in the list
# + Matches between one and unlimited times
# | Or
start = time.time()
matches = re.findall("[\w']+", data)
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("Tokenization with regular expression: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


#------------WordTokenisation using NLTK---------

start = time.time()
nltk_tw=word_tokenize(data)
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("Tokenization using NLTK : {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

#------------WordTokenisation using spacy---------

nlp = English()
nlp.max_length = 177418665
my_doc = nlp(data)
token_list = []
start = time.time()
for token in my_doc:
    token_list.append(token.text)
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("Tokenization using spacy: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


#------------WordTokenisation using Keras---------
start = time.time()
result = text_to_word_sequence(data)
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("Tokenisation using Keras: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

#------------WordTokenization using Gensim---------
start = time.time()
text= list(tokenize(data))
number_of_words = len(text)
print("Number of words:", number_of_words)
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("Tokenisation using Gensim: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


tokenizer = BertTokenizer.from_pretrained(data)
