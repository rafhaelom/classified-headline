import streamlit as st

import numpy as np
import pandas as pd

import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import spacy

nlp = spacy.load('pt_core_news_sm')

# vectorizer models
bow_vectorizer = pickle.load(open('model\bow_vectorizer.pkl', "rb"))
tfidf_vectorizer = pickle.load(open('model\tfidf_vectorizer.pkl', "rb"))

# models
model_nb_bow = pickle.load(open('model\model_nb_bow.pkl', "rb"))
model_nb_tfidf = pickle.load(open('model\model_nb_tfidf.pkl', "rb"))
model_rf_bow = pickle.load(open('model\model_rf_bow.pkl', "rb"))
model_rf_tfidf = pickle.load(open('model\model_rf_tfidf.pkl', "rb"))

# funções de normalização do texto
def sentence_tokenizer(sentence):
  return [token.lemma_ for token in nlp(sentence.lower()) 
              if (token.is_alpha & ~token.is_stop)]

def normalizer(sentence):
  tokenized_sentence = sentence_tokenizer(sentence)
  return ' '.join(tokenized_sentence)

model.predict(vec.transform([normalizer('Por que veículos usados por deputados não são identificados?')]))