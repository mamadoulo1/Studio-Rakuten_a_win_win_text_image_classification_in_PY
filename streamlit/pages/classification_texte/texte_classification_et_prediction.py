#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 20:04:56 2022

@author: haeji
"""

import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os
import time
import gensim
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from numpy import array
from keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten, GlobalAveragePooling1D
from keras.utils import np_utils
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm.notebook import tqdm
from PIL import Image

# gestion des différents paths 
def csvpath(script_dir, csv):
    return script_dir + os.path.normpath('\\' + csv)
def imgpath(script_dir, img):
    return script_dir + os.path.normpath('\\' + img)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache
def process(data_train):
    data_train['designation_str_tokens'] = ""
    for i in (range(len(data_train))):
        data_train['designation_str_tokens'].iloc[i] += "".join(data_train['designation_token_final'].iloc[i])
    data_train['designation_str_tokens'] = data_train['designation_str_tokens'].map(lambda x: re.sub(r',', '', x))
    data_train['designation_str_tokens'] = data_train['designation_str_tokens'].map(lambda x: re.sub(r"'", '', x))
    X_data_train = data_train.designation_str_tokens
    X_data_train = X_data_train.str.replace("[", "").str.replace("]", "")
    return X_data_train

@st.cache(allow_output_mutation=True)
def vectorize(vectorizer_choisi,X_train,X_test):
    if vectorizer_choisi == "TF-IDF Vectorizer":
        vectorizer = TfidfVectorizer()
    elif vectorizer_choisi == "COUNT Vectorizer":
        vectorizer = CountVectorizer()
    vectorizer.fit(X_train)
    X_train_vc = vectorizer.transform(X_train)
    X_test_vc = vectorizer.transform(X_test)
    return X_train_vc, X_test_vc

@st.cache
def lr(vectorizer_choisi,X_train,X_test,y_train,y_test):
    lr = LogisticRegression()
    lr.fit(vectorize(vectorizer_choisi,X_train,X_test)[0], y_train)
    y_pred_lr = lr.predict(vectorize(vectorizer_choisi,X_train,X_test)[1])
    return classification_report(y_test, y_pred_lr), confusion_matrix(y_test, y_pred_lr)

#@st.cache
@st.experimental_singleton(suppress_st_warning=True)
def pred(_model,X_test_deep,y_test_deep):
    test_pred = _model.predict(X_test_deep)
    test_pred_class = np.argmax(test_pred, axis=1)
    return classification_report(y_test_deep, test_pred_class)


#@st.cache ne fonctionne pas
@st.experimental_singleton(suppress_st_warning=True)
def history(_model,X_train_deep,y_train_deep):
    return _model.fit(X_train_deep, y_train_deep, batch_size=100, epochs=11,validation_split=0.2)

@st.cache
def lr2(X_train_combi,y_train_combi,X_test_combi,y_test_combi):
    lr2= LogisticRegression()
    lr2.fit(X_train_combi, y_train_combi)
    y_pred_lr2 = lr2.predict(X_test_combi)
    return classification_report(y_test_combi, y_pred_lr2)

def write():
    """Used to write the page in the app.py file"""

# =============================================================================
#  CLASSIFICATION DE TEXTE
# =============================================================================
    st.markdown("<h1 style='color: red'>CLASSIFICATION DE TEXTE</h1>", unsafe_allow_html=True)
    st.markdown("***")

    st.write("Pour la partie texte, nous avons testé trois approches :")
    st.write("- Machine Learning Classique")
    st.write("- Deep Learning")
    st.write("- Combinaison de Machine Learning & Deep Learning")


#------------------------------------------------------------------------------

### La base des données
    st.text("")
    st.text("")
    st.markdown("<h2 style='color: red'>Aperçu de base des données après préprocessing</h2>", unsafe_allow_html=True)
    st.write("Ci-dessous nous retrouvons la base de données prête pour le Machine Learning")

    data_train = pd.read_csv(csvpath(SCRIPT_DIR,'X_train_rakuten_afterEDA_preprocessing.csv'))
    #data_train = pd.read_csv('X_train_rakuten_afterEDA_preprocessing.csv')

    list_class = [10, 2280 ,  50 ,1280 ,2705, 2522, 2582, 1560, 1281, 1920, 2403, 1140, 2583, 1180, 1300 ,2462, 1160, 2060 , 40,   60 ,1320 ,1302 ,2220 ,2905, 2585, 1940 ,1301]       
    list_class_onehot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]


    X = process(data_train)
    y = data_train.prdtypecode
    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state = 0)

    left1, right1 = st.columns([2,1])

    left1.markdown("<h6 style='text-align: center'>X_train</h6>", unsafe_allow_html=True)
    left1.dataframe(data=X_train.head())

    right1.markdown("<h6 style='text-align: center'>y_train</h6>", unsafe_allow_html=True)
    right1.dataframe(data=y_train.head())

    st.write("La dimension de X_train:", X_train.shape)
    st.write("La dimension de y_train:", y_train.shape)
    st.write("La dimension de X_test:", X_test.shape)
    st.write("La dimension de y_train:", y_test.shape)


#------------------------------------------------------------------------------

### Machine Learning Classique
    st.text("")
    st.text("")
    st.markdown("<h2 style='color: red'>Machine Learning Classique</h2>", unsafe_allow_html=True)


### Vectorisation des données
    st.text("")
    st.subheader("Vectorisation des données")
    st.write("Avant d’appliquer les algorithmes de machine learning, nous devons vectoriser le texte afin de le transformer en données numériques interprétables par l’algorithme.")
    st.write("Parmi les différrents méthodes de vectorisation, nous avons choisi :")
    st.write("- CountVectorizer : représentation de fréquence des mots dans le texte")
    st.write("-	TFIDF Vectorizer : représentation de fréquence et de poids (importance) dans le texte")

    vectorizer_choisi = st.selectbox(label="Choix de Vectorisateur", options=["TF-IDF Vectorizer","COUNT Vectorizer"])

  
    st.markdown("<h6 style='text-align: center'>X_train après la vectorisation</h6>", unsafe_allow_html=True)
    st.write(vectorize(vectorizer_choisi,X_train,X_test)[0])

    st.write("La dimension de X_train:", vectorize(vectorizer_choisi,X_train,X_test)[0].shape)
    st.write("La dimension de X_test:", vectorize(vectorizer_choisi,X_train,X_test)[1].shape)


## Création des classifieurs
    st.text("")
    st.subheader("Classifieurs")
    st.write("Les quatre algorithmes classiques utilisés sont Régression Logistique, SVM, Random Forest, et puis Voting Classifier qui regroupe les trois algorithmes.")


#------------------------------------------------------------------------------

### Deep Learning
    st.text("")
    st.text("")
    st.markdown("<h2 style='color: red'>Deep Learning framework tensorflow/keras</h2>", unsafe_allow_html=True)
    st.write("nous avons utilisé un modèle élémentaire baseline")


## Définition features & labels
    st.text("")
    st.subheader("One-hot encoding")
    st.write("Nous avons appliqué one-hot encoding sur notre base de données afin d'obtenir les données adaptées pour le Deep Learning ")

    data_train['onehot']= 0
    for i in range(len(data_train)):
        index = list_class.index(data_train['prdtypecode'].iloc[i])
        data_train['onehot'].iloc[i] = list_class_onehot[index]

    y = data_train.onehot
    y = np.array(y)
    X_encoded = [one_hot(i, 1000) for i in X]
    X_padded = pad_sequences(X_encoded, 1000, padding='post')

    X_train_deep, X_test_deep, y_train_deep, y_test_deep = train_test_split(X_padded, y , test_size=0.2)

    left2, right2 = st.columns([3,1])

    left2.markdown("<h6 style='text-align: center'>X_train</h6>", unsafe_allow_html=True)
    left2.dataframe(data=X_train_deep[:5])

    right2.markdown("<h6 style='text-align: center'>y_train</h6>", unsafe_allow_html=True)
    right2.dataframe(data=y_train_deep[:5])

    st.write("La dimension de X_train:", X_train_deep.shape)
    st.write("La dimension de y_train:", y_train_deep.shape)
    st.write("La dimension de X_test:", X_test_deep.shape)
    st.write("La dimension de y_train:", y_test_deep.shape)


## Création du modèle
    st.text("")
    st.subheader("Modèle Deep Learning baseline")
    st.write("Pour la première tentative, seulement une couche Embedding avec une petite taille de vocabulaire à 1000 et un vecteur dense de sortie de 750, puis une couche classifieur dense et un entrainement sur 11 epochs")

    model = Sequential()
    model.add(Embedding(1000, 750))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(27, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    summary = model.summary()
    img5 = Image.open(imgpath(SCRIPT_DIR, 'Picture5.png'))
    st.image(img5)
    #st.image("Picture5.png")




#------------------------------------------------------------------------------

### Combi Deep Learning + Machine Learning Classique
    st.text("")
    st.text("")
    st.markdown("<h2 style='color: red'>Combinaison Machine Learning Classique + Deep Learning</h2>", unsafe_allow_html=True)
    st.write("Pour la combinaison de Machine Learning Classique et Deep Learning,")
    st.write("-	Nous avons utilisé Doc2Vec pour générer un vecteur de taille 100 sur chaque texte")
    st.write("-	Puis appliqué nos algorithmes de machine learning sur ces features vectorisées")


## Création du modèle Deep
    st.text("")
    st.subheader("Modèle Doc2Vec")
    st.text("Voici les échantillons que nous obtenons après l'application du modèle Doc2Vec")
    train_tagged = [TaggedDocument(words=nltk.word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in tqdm(enumerate(X))]
    model_combi = Doc2Vec(vector_size=100, epochs=10)
    model_combi.build_vocab(train_tagged)
    model_combi.train(train_tagged, total_examples=model_combi.corpus_count, epochs=model_combi.epochs)

    X = [i.split(" ") for i in X]
    X = [model_combi.infer_vector(phrase) for phrase in X]
    X = np.array(X)
    y = data_train.onehot
    y = np.array(y)
    X_train_combi, X_test_combi, y_train_combi, y_test_combi = train_test_split(X, y , test_size=0.2)

    left3, right3 = st.columns([3,1])

    left3.markdown("<h6 style='text-align: center'>X_train</h6>", unsafe_allow_html=True)
    left3.dataframe(data=X_train_combi[:5])

    right3.markdown("<h6 style='text-align: center'>y_train</h6>", unsafe_allow_html=True)
    right3.dataframe(data=y_train_combi[:5])


## Machine Learning Classique
    st.text("")
    st.subheader("Machine Learning Classique")
    st.text("Nous avons ensuite appliqué le machine learning classique en passant")
    st.text("les échantillons transformés par l'application de deep learning Doc2Vec.")



# =============================================================================
# PRÉDICTION DE TEXTE
# =============================================================================
    st.markdown("<h1 style='color: red'>PRÉDICTION DE TEXTE</h1>", unsafe_allow_html=True)
    st.markdown("***")

    st.write("Nous avons les résultats des trois approches que nous avons testé:")
    st.write("- Machine Learning Classique")
    st.write("- Deep Learning")
    st.write("- Combinaison de Machine Learning & Deep Learning")


#------------------------------------------------------------------------------
### Machine Learning Classique
    st.text("")
    st.text("")
    st.markdown("<h2 style='color: red'>Machine Learning Classique</h2>", unsafe_allow_html=True)

    st.write("Nous constatons qu’en général, TFIDF Vectorizer est plus performant de 2% en terme de précision. Sa performance est plus marqué avec SVM qui présente une différence de 10% de précision en contrepartie au temps d’exécution plus long.")
    st.write("Par la suite, nous allons garder TFIDF Vectorizer")
    st.write("Parmi les algorithmes, SVM et VotingClassifier restent les plus performanants avec une précision de 0,81. Logistique Régression a une performance similaire avec une précision de 0,80. Par contre nous remarquons un temps d’exécution énorme avec SVM et VotingClassifier.")
    st.write("Le bon compromis serait de garder Logistique Régression qui donne une performance similaire avec un temps d’exécution plus rapide.")


    exp8 = st.expander("Logistic Regression")
    exp8.code(lr(vectorizer_choisi,X_train,X_test,y_train,y_test)[0])

#exp9 = st.expander("SVC")
#exp9.code(svc()[0])

#exp10 = st.expander("Random Forest Classifier")
#exp10.code(rf()[0])

#exp11 = st.expander("Voting Classifier")
#exp11.code(vc()[0])

    st.write("Les classes les mieux prédites et les moins bien prédites restent les mêmes dans les différents algorithmes essayés :")
    st.write("-	Les mieux prédites sont les classes 2583 et 2905")
    st.write("-	Les moins bien prédites sont les classes 10 et 1281")
    st.write("Les prédictions de la classe 10 :")
    st.write("Les fausses prédictions fréquentes sont les classes 2280, 2403, 2705. On remarque la similitude des classes, toutes liées à la lecture (10 éditions , 2280 magazines, 2403 livres_par_lot, 2705 livres)")
    st.write("Les prédictions de la classe 1281 :")
    st.write("Les fausses prédictions fréquentes sont la classe 1280. On remarque la similitude des deux classes, tous les deux liées au jeu d’enfants (1281 jeux_enfant , 1280 jouets_enfant)")
    
    exp16 = st.expander("Logistic Regression")
    matrix_lr = lr(vectorizer_choisi,X_train,X_test,y_train,y_test)[1]
    fig_lr, ax_lr = plt.subplots()
    sns.heatmap(matrix_lr, ax=ax_lr, annot=True, cmap="coolwarm")
    plt.xticks(np.arange(len(list_class)), list_class)
    plt.yticks(np.arange(len(list_class)), list_class,rotation='horizontal')
    plt.ylabel('Classe réelle')
    plt.xlabel('Classe prédite')
    exp16.write(fig_lr)

#exp17 = st.expander("SVC")
#matrix_svc = svc()[1]
#fig_svc, ax_svc = plt.subplots()
#sns.heatmap(matrix_svc, ax=ax_svc, annot=True, cmap="coolwarm")
#plt.xticks(np.arange(len(list_class)), list_class)
#plt.yticks(np.arange(len(list_class)), list_class,rotation='horizontal')
#plt.ylabel('Classe réelle')
#plt.xlabel('Classe prédite')
#exp17.write(fig_svc)

#exp18 = st.expander("Random Forest Classifier")
#matrix_rf = rf()[1]
#fig_rf, ax_rf = plt.subplots()
#sns.heatmap(matrix_rf, ax=ax_rf, annot=True, cmap="coolwarm")
#plt.xticks(np.arange(len(list_class)), list_class)
#plt.yticks(np.arange(len(list_class)), list_class,rotation='horizontal')
#plt.ylabel('Classe réelle')
#plt.xlabel('Classe prédite')
#exp18.write(fig_rf)

#exp19 = st.expander("Voting Classifier")
#matrix_vc = vc()[1]
#fig_vc, ax_vc = plt.subplots()
#sns.heatmap(matrix_vc, ax=ax_vc, annot=True, cmap="coolwarm")
#plt.xticks(np.arange(len(list_class)), list_class)
#plt.yticks(np.arange(len(list_class)), list_class,rotation='horizontal')
#plt.ylabel('Classe réelle')
#plt.xlabel('Classe prédite')
#exp19.write(fig_vc)


#------------------------------------------------------------------------------
### Deep Learning
#    st.text("")
#    st.text("")
#    st.markdown("<h2 style='color: red'>Deep Learning baseline</h2>", unsafe_allow_html=True)

#   st.write("Vu le grand nombre de mots de notre dictionnaire qui est à l’ordre de 48.000, le modèle ne prédit pas correctement et donne une précision de 0.6 et il y a une grande disparité de précision entre les classes. ")
#    st.write("Il faut donc augmenter significativement la taille de vocabulaire de la couche embeddding (1000 à 48000) pour les prochains entraînements.")


## Application
#    history_deep = history(model,X_train_deep,y_train_deep)

## Évaluation
#    loss, accuracy = model.evaluate(X_train_deep, y_train_deep, verbose=0)
#    st.write("Accuracy:", accuracy)

## Prédiction
#    st.code(pred(model,X_test_deep,y_test_deep))


## Affichage graphique de l'évolution d'entrainement
#    st.write("Le graphique ci-dessous affiche la précision des échantillons. La précision de l’échantillon de validation reste très proche de celle de l’entraînement, donc il n’y a pas de problème de surapprentissage.")
#    accuracy = pd.DataFrame({'accuracy_train':history_deep.history['accuracy'],'accuracy_validation':history_deep.history['val_accuracy']})

#    choice = st.multiselect("Choix des échantillons", accuracy.columns.tolist(),default="accuracy_train")
#    accuracy_data = accuracy[choice]
#    st.markdown("<h6 style='text-align: center'>La précision</h6>", unsafe_allow_html=True)
#    st.line_chart(accuracy_data)


#------------------------------------------------------------------------------
### Combi Deep Learning + Machine Learning Classique
    st.text("")
    st.text("")
    st.markdown("<h2 style='color: red'>Combinaison Machine Learning Classique + Deep Learning</h2>", unsafe_allow_html=True)

    st.write("La combinaison de machine learning + deep learning est beaucoup moins performant que le machine learning classique malgré son temps d’exécution plus court.")


    exp12 = st.expander("feature extraction avec doc2Vec puis Logistic Regression")
    exp12.code(lr2(X_train_combi,y_train_combi,X_test_combi,y_test_combi))

#exp13 = st.expander("SVC")
#exp13.code(svc2())

#exp14 = st.expander("Random Forest Classifier")
#exp14.code(rf2())

#exp15 = st.expander("Voting Classifier")
#exp15.code(vc2())