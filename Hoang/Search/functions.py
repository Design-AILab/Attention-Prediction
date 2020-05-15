import pandas as pd
import numpy as np
import string

data = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vS3CYy0jW1weK1x_dbLz6EGCqPgWXWWmaZHEV-nzj2Gb5sjrOMsjiKObee-_r6KVTZkBMEiVQEGdgwS/pub?gid=529557672&single=true&output=csv")

# Extract word vectors
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

def prep(x):
  x = str(x)
  translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space
  x = x.translate(translator)
  return x

desc_list = np.load("description.npy")
tags_list = np.load("tag.npy")


def search(number_of_output = 15):
  search_term = input("Search term: ")

  search_term = prep(search_term)

  if len(search_term) != 0:
    v = sum([word_embeddings.get(w, np.zeros((100,))) for w in search_term.split()])/(len(search_term.split())+0.001)
  else:
    v = np.zeros(100)
  Y = np.sum((desc_list - v)**2, axis = 1) + np.sum((tags_list - v)**2, axis = 1)

  X = list(range(len(data)))

  Z = [x for _,x in sorted(zip(Y,X))]

  if type(number_of_output) != int:
    print("Error input")
  return data.url[Z[:number_of_output]]

