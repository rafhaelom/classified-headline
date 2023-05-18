import streamlit as st

from goose3 import Goose

import numpy as np
import pandas as pd

import pickle

# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import metrics

import spacy

nlp = spacy.load('pt_core_news_sm')

# vectorizer models
bow_vectorizer = pickle.load(open('model/bow_vectorizer.pkl', "rb"))
tfidf_vectorizer = pickle.load(open('model/tfidf_vectorizer.pkl', "rb"))

# models
model_nb_bow = pickle.load(open('model/model_nb_bow.pkl', "rb"))
model_nb_tfidf = pickle.load(open('model/model_nb_tfidf.pkl', "rb"))
# model_rf_bow = pickle.load(open('model/model_rf_bow.pkl', "rb"))
# model_rf_tfidf = pickle.load(open('model/model_rf_tfidf.pkl', "rb"))

# funções de normalização do texto
def sentence_tokenizer(sentence):
  return [token.lemma_ for token in nlp(sentence.lower()) 
              if (token.is_alpha & ~token.is_stop)]

def normalizer(sentence):
  tokenized_sentence = sentence_tokenizer(sentence)
  return ' '.join(tokenized_sentence)

##### INÍCIO APP.
st.markdown("""<h1 align='center'>Classificação da notícia pela manchete 📰<h1 align='justify'>""", unsafe_allow_html=True)

# Opção para a origem da notícia
opcao = st.selectbox("🎲 Escolha um opção:", ["Texto", "Link"], index=0, help="Opção de qual origem é a notícia, se é o título da notícia ou um link para extrair o título da notícia.")
if opcao == "Texto":
  form = st.form(key='my_form')
  texto_manchete = form.text_area(label="📝 Insira uma manchete: ", value="", placeholder='Insira a manchete aqui...')
  submit_button = form.form_submit_button(label='Resumir 🎉')
if opcao == "Link":
  form = st.form(key='my_form')
  url_noticia = form.text_input(label="🌐 Insira um link: ", value="", placeholder='Informe o link aqui...')
  submit_button = form.form_submit_button(label='Classificar 🎉')

# Classificador da manchete após a escolha da origem
if submit_button and opcao == "Texto" and texto_manchete != "":
  with st.spinner('Classificando...'):
    predicao_bow_nb = model_nb_bow.predict(bow_vectorizer.transform([normalizer(texto_manchete)]))
    predicao_tfidf_nb = model_nb_tfidf.predict(tfidf_vectorizer.transform([normalizer(texto_manchete)]))

    st.write("### Classificador Naive Bayes 🧮")
    st.success(body=f"Bag of Words: {predicao_bow_nb}", icon="✅")
    st.success(body=f"TF-IDF: {predicao_tfidf_nb}", icon="✅")
if submit_button and opcao == "Link" and url_noticia != "":
  with st.spinner('Extraindo manchete...'):
    g = Goose()
    article = g.extract(url=url_noticia)
    manchete_link = article.title
    g.close()

    st.info(body=manchete_link)
  with st.spinner('Classificando...'):
    predicao_bow_nb = model_nb_bow.predict(bow_vectorizer.transform([normalizer(manchete_link)]))
    predicao_tfidf_nb = model_nb_tfidf.predict(tfidf_vectorizer.transform([normalizer(manchete_link)]))

    st.write("### Classificador Naive Bayes 🧮")
    st.success(body=f"Bag of Words: {predicao_bow_nb}", icon="✅")
    st.success(body=f"TF-IDF: {predicao_tfidf_nb}", icon="✅")
else:
  st.warning(body="Insira uma manchete!!!", icon="⚠")


st.write("by Rafhael Martins")