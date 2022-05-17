#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:01:13 2022

@author: haeji
"""

import os
import time
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from PIL import Image

# gestion des différents paths 
def csvpath(script_dir, csv):
    return script_dir + os.path.normpath('\\' + csv)
def imgpath(script_dir, img):
    return script_dir + os.path.normpath('\\' + img)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# EXPLORATION DES DONNÉES
# =============================================================================


def write():   

    st.title("EXPLORATION DES DONNÉES")
    st.markdown("***")

    st.write("Dans la phase d'exploration, nous avons effectué une analyse des données, puis nous avons appliqué les préprocessing pour obtenir une base de données prête pour le Machine Learning.")

#------------------------------------------------------------------------------

### Caractéristiques des données ###
    st.text("")
    st.text("")
    st.header("Caractéristiques de la base des données")


## Aperçu des données
    st.text("")
    st.subheader("Aperçu général")

# Les données texte
    st.write("La base de données est composée de l'ensemble d'apprentissage et de l'ensemble de test. L'ensemble d'apprentissage comporte les variables explicatives 'X_train' et la variable cible 'y_train'. L'ensemble de test comporte que les variables explicatives 'X_test'.")
    st.write("Nous avons quatre variables explicatives :\n"
         "- 'designation' : titre et description du produit\n"
         "- 'description' : description détaillée du produit\n"
         "- 'productid' : identifiant du produit\n"
         "- 'imageid'' : identifiant de l'image\n")
    st.write("Le variable cible 'prdtypecode' correspond au code associé à la catégorie dont chaque produit appartient.")
    st.write("Ci-dessous est l'aperçu de l'ensemble d'apprentissage :")

    left, right = st.columns([3,1])

    data_train = pd.read_csv(csvpath(SCRIPT_DIR,'X_train_update.csv'),index_col=0)
    left.markdown("<h6 style='text-align: center'>X_train</h6>", unsafe_allow_html=True)
    left.dataframe(data=data_train.head())

    target = pd.read_csv(csvpath(SCRIPT_DIR,'Y_train_CVw08PX.csv'),index_col=0)
    right.markdown("<h6 style='text-align: center'>y_train</h6>", unsafe_allow_html=True)
    right.dataframe(data=target.head())

    data_test = pd.read_csv(csvpath(SCRIPT_DIR,'X_test_update.csv'),index_col=0)

    st.write("La dimension de X_train:", data_train.shape)
    st.write("La dimension de y_train:", target.shape)
    st.write("La dimension de X_test:", data_test.shape)

# Les données image
    st.text("")
    st.write("Nous avons aussi les données images associées à chaque produit.\n"
         "La taille de chaque image est comprise entre 3ko et 105ko. Le nombre d'images avec une taille inférieure à 50ko est de 45.726, et celui au-dessus est de 38.650.\n")
    st.write("Ci-dessous ,quelques exemples des images téléchargées : ")
    st.markdown("<h6 style='text-align: center'>Données images</h6>", unsafe_allow_html=True)
    st.image(Image.open(imgpath(SCRIPT_DIR,"Picture0.png")), width =700)

    st.text("")
    st.write("Suivi de la règle de nomenclature des images fournies, nous avons créé une nouvelle colonne 'nom_img' au dataframe de référence en lieu et place des deux champs 'productid' et 'imageid'.")
    data_train = pd.read_csv(csvpath(SCRIPT_DIR,'X_train_rakuten_afterEDA_preprocessing.csv'))
    exp2 = st.expander("Nomenclature des images")
    exp2.table(data_train["nom_img"].head())



## Les classes cibles
    st.text("")
    st.subheader("Les classes cibles")

    st.write("Le nombre des classes cible :",len(target['prdtypecode'].unique()))
    st.write("La liste des classes :", target.value_counts().index)

    st.write("Les classes sont sous forme de nombres ce qui sera très utile pour les algorithmes. Néanmoins, pour être plus parlant, nous avons donné un nom représentatif à chacune de ces classes.\n"
         "Le nom a été choisi en regardant des dizaines d'images de chaque classe, le nuage de mots et l'histogramme de fréquences des tokens extraits des variables 'designation' et 'description'.")
    st.write("Ci-dessous est la liste des noms que nous avons donné à chaque classe : ")

    exp3 = st.expander("Nom des classes")
    noms_classe = pd.DataFrame({'prdtypecode':[10,2280,50,1280,2705,2522,2582,1560,1281,1920,2403,1140,2583,1180,1300,2462,1160,2060,40,60,1320,1302,2220,2905,2585,1940,1301],'nom de classe':['editions', 'magazines','equipment_playstation','jouets_enfant','livres','fournitures_scolaire','materiel_jardin','mobilier_interieur','jeux_enfants','librairie','livres_par_lot','figurine','materiel_piscine','univers_de_fiction','equipements_telecommandes','materiel_jeux_video','jeux_de_cartes_collection','decoration','jeux_video','console_de_jeu','materiel_bebe','peche_lampe_plein_air','materiel_animaux_compagnie','jeux_sur_PC','accessoires_outil','articles_nourriture','vetements_nouveau_ne_et_billard_flechettes']})
    exp3.table(data=noms_classe)

    st.write("La base des données est déséquilibrée avec la classe 2583 qui est surreprésentée avec 12%, 17 classes qui représentent chacune entre 3-6%, et 9 petites classes qui représentent moins de 2% chacune. \n"
         "Ainsi, le TOP 10 des classes sur 27 équivaut à 64.5% de la base des données.")
    st.write("Ce découpage peut se faire en phrases, en mots voire même en caractères selon l'utilité dans le cas expérimenté.")
    st.markdown("<h6 style='text-align: center'>Distribution des classes</h6>", unsafe_allow_html=True)
    classe = pd.DataFrame(np.round(target['prdtypecode'].value_counts(normalize=True)*100))
    st.bar_chart(classe)


#------------------------------------------------------------------------------
### Préprocessing ###
    st.text("")
    st.text("")
    st.header("Préprocessing des données")


## Préprocessing du texte
    st.text("")
    st.subheader("Préprocessing du texte")

# tokenisation
    st.write("Tokenisation")
    st.write("La tokenisation est l'opération qui consiste à segmenter une phrase en unités 'atomiques': les tokens. Ce découpage se faisant en phrases, en mots voire même en caractères selon l'utilité dans le cas expérimenté")
    st.write("Nous avons appliqué la tokenisation afin d'obtenir le découpage en mots.")

    tokenisation = pd.DataFrame({'designation':data_train['designation'][:0], 'tokenization':data_train['designation'][:0].apply(lambda x: word_tokenize(x))})
    exp4 = st.expander("Exemple de tokenisation")
    label, texte = exp4.columns([1,4])
    label.write("texte original :")
    texte.write(data_train['designation'][0])
    label.write("texte tokenisé :")
    texte.markdown(word_tokenize(data_train['designation'][0]))


# stopwords
    st.write("Stopwords")
    st.write("Nous avons appliqué un filtre qui supprime les mots outils 'stopwords' qui ne portent pas d'information de sens pour aider à classifier chaque produit")

    fr_stopwords = set(stopwords.words("french"))
    fr_stopwords.update([":", "/","//","(",")","N°","n°","%","?","+",".","&","[","]","*","''","``", "'", "////", "br/"])
    fr_stopwords.update(["°","@","xcm","-cm","kg/m","kg/","g/m","m³/h","m²","gr/m²","_","#",";","ø","--","²",
                     "_-","s'en","s'il","n'est","d'un","d'une","...","g/m²","m/h","c'est","qu'il","qu'elle","jusqu"])
    list_stopwords_colours = ["rose","gris","blanc","noir","vert","bleu","rouge","marron","black","anthracite","jaune"]
    list_stopwords_adjectives = ["bon","nouveau","facile","rare","commune","grand","petit","mini","deux","ronde","bas","chaud",
                             "pliable","neuf","ovale"]
    list_stopwords_adjectivesbis = ["doux","confortable","new","nouvelle","nouveaux","grande","complet","integrale","legere"]
    list_stopwords_adverb = ["afin","enfin","chaque","plus","contre","dont","sans","pcs","comment","plusieurs",
                         "sous","chez","entre","tous","depuis","aussi","alors"]
    list_stopwords_adverbis = ["egalement","tout","toutes","comme","ainsi","assez","jamais","encore","lorsque",
                           "tres","toujours","apres","quand","grace"]
    list_stopwords_adverbter = ["particulierement","beaucoup", "suffisamment","pendant","certainement",
                            "immediatement","seulement","doucement","neanmoins","meme", "avant","environ",
                            "peu","propos","legerement","aujourd'hui","trop","souvent"]
    list_stopwords_various_names = ["necessaire","facilement","non","peut","ans","mois","attention"]
    list_stopwords_various_namesbis = ["description","comprend","qualite","haute","merci","ci-dessus","caracteristiques","couleur","couleurs"]
    list_stopwords_various_namester = ["annee","annees","taille","inclus","dimensions","poids","difference","different"]
    list_stopwords_verbs = ["etre","avoir","donnons","vendons","reversons","peuvent","pouvez","gardez",
                        "permettre","n'existe","utiliser","utilise","fait","plait","parlait"]
    list_stopwords_pronouns = ["cette", "cet", "leurs", "ceux", "toute"]
    fr_stopwords.update(list_stopwords_colours)
    fr_stopwords.update(list_stopwords_adjectives)
    fr_stopwords.update(list_stopwords_adjectivesbis)
    fr_stopwords.update(list_stopwords_adverb)
    fr_stopwords.update(list_stopwords_adverbis)
    fr_stopwords.update(list_stopwords_adverbter)
    fr_stopwords.update(list_stopwords_various_names)
    fr_stopwords.update(list_stopwords_various_namesbis)
    fr_stopwords.update(list_stopwords_various_namester)
    fr_stopwords.update(list_stopwords_verbs)
    fr_stopwords.update(list_stopwords_pronouns)

    exp5 = st.expander("Notre liste des stopwords")
    exp5.write(fr_stopwords)

# lemmatisation
    st.write("Lemmatisation à l'aide du pipeline préentrainé fr_core_news_lg de la librairie spacy")
    st.write("La lemmatisation consiste à représenter les mots sous leur forme canonique : féminin devenant masculin, pluriel en singulier, verbe conjugé en infinitif. Nous avons adopté cette transformation car il y a eu peu d'effets de bords indésirables et cette technique a diminué le volume du dictionnaire Bag of Words.")

    exp6 = st.expander("Exemple de lemmatisation")
    exp6.image(Image.open(imgpath(SCRIPT_DIR,"lemmatisation.png")))
    #lemmatizer=WordNetLemmatizer()
    #lemme = pd.DataFrame({'texte originale':word_tokenize(data_train['designation'][3])})
    #lemme['texte lemmatisé'] = lemme['texte originale'].map(lambda x: lemmatizer.lemmatize(x))
    #exp6.write(lemme)

# suppression
    st.write("Suppression") 
    st.write("Autre technique qui a diminué significativement le dictionnaire BOW, la suppression de tous les mots qui n'étaient présents qu'une seule fois dans leur classe.")
    st.write("Pour compenser un petit effet de bord induit, nous avons alors du supprimer 303 observations (produits du dataset) qui n'avaient plus de texte une fois appliqué ce filtre.")


## Préprocessing de l'image
    st.text("")
    st.subheader("Préprocessing de l'image")

    st.write("En Data Science, travailler sur des problématiques utilisant des images implique de savoir comment faire des transformations sur celles-ci en amont du  traitement d'un modèle. Ces techniques ont pour but d'essentialiser l'information, de retravailler l'image pour obtenir une meilleure compréhension de la part de votre algorithme.\n")
    st.write("Dans notre projet, toutes les images comportent des bords blancs que nous allons supprimer.\n")
    st.write("Nous avons supprimé 237 articles qui étaient associés à des fichiers image de taille trop petite 2 Kb")

    exp7 = st.expander("Quelques exemples")
    exp7.write("- Image de taille nulle aprés l'opération de cropping (effet du seuil de blanc mis à 192 sur 255)")
    exp7.image(Image.open(imgpath(SCRIPT_DIR,"Picture1.png")))
    exp7.text("")
    exp7.write("- Très petites images : les stylos")
    exp7.image(Image.open(imgpath(SCRIPT_DIR,"Picture2.png")))
    exp7.text("")
    exp7.write("- Images incompréhensibles sans sa description textuelle")
    exp7.image(Image.open(imgpath(SCRIPT_DIR,"Picture3.png")))
    exp7.image(Image.open(imgpath(SCRIPT_DIR,"Picture4.png")))


#------------------------------------------------------------------------------
### Les données finales
    st.text("")
    st.text("")
    st.header("Les données finales")
    st.write("Après notre étape de préprocessing, nous obtenons la base des données prêt pour le Machine Learning.\n")
    st.write("Elle comporte 4 colonnes : \n"
         "- 'designation' : titre et description du produit en francais\n"
         "- 'designation_token_final : liste des tokens déduits de 'designation'\n"
         "- 'prdtypecode : classe d'appartenance du produit (cible à prédire) \n"
         "- 'nom_img : image du produit \n")
    st.markdown("<h6 style='text-align: center'>Base des données finale</h6>", unsafe_allow_html=True)
    st.dataframe(data=data_train)
    st.write("La dimension des données:", data_train.shape)


