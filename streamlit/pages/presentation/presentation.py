import streamlit as st
import os
from PIL import Image

# gestion des différents paths 
def imgpath(script_dir, img):
    return script_dir + os.path.normpath('\\' + img)

def write():
    """appelé depuis le script principal"""

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    col1, col2, col3 = st.columns([1,2,1])

    with col1:
        st.write("")

    with col2:
        logo_DS = Image.open(imgpath(SCRIPT_DIR, 'logo_datascientest.png'))
        st.image(logo_DS,caption='Institut de formation en datascience')
        

    with col3:
        st.write("")

    st.title("Présentation du projet")
    st.header("Contexte")
    st.text(
        "Rakuten Institute of Technology (RIT) ,département de recherche et innovation,\n"
        "a déposé sur le site challengedata de l'ENS un challenge sur l'un des éléments clé\n"
        "du coeur de métier de Rakuten qui est de mettre à disposition des millions\n"
        "d'articles classés correctement dans l'une des 1500 catégories de leur plate-forme\n"
        "de e-commerce l'une des plus importantes au niveau mondial avec ses quelques\n"
        "1.3 milliards de membres." )

    st.header("Objectifs du projet et la méthodologie suivie")
    st.text(
        "Le challenge intitulé\n"
        "Rakuten France Multimodal Product Data Classification\n"
        "propose de travailler sur un périmétre bien plus réduit avec un jeu\n"
        "de données d'entrainement de 84916 produits répartis en 27 classes.\n"
        "La métrique de performance weighted-F1 score est imposée.\n"
        "\n"
        "Ce projet fil rouge mené dans le cadre de la formation DS de Datascientest\n"
        "a pour but de nous construire une première expérience tangible d'un cycle\n"
        "de développement d'un modèle de Machine Learning ou de Deep Learning.\n"
        "\n"
        "Une fois téléchargées les données du site du challenge ENS,\n"
        "nous avons effectué une phase d'analyse des données , appliqué les préprocessing\n"
        "jugés nécessaires pour préparer les données texte et image aux premiers modèles\n"
        "de ML choisis au sein des différentes familles d'algorithmes de classification\n"
        "ayant des approches sous jacentes complémentaires.\n"
        "\n"
        "Ainsi,nous avons pu batir un premier socle de résultats communèment appelé baseline\n"
        "Une fois construit ce premier jeu de performances brutes pour chaque algorithme\n"
        "candidat, nous avons effectué plusieurs cycles d'itérations nous permettant\n"
        "d'expérimenter de nombreuses techniques de NLP appliquées au texte et ainsi batir\n"
        "un tableau de résultats solides nous permettant finalement de choisir \n"
        "deux modèles convaincants pour cette présentation."
    )
