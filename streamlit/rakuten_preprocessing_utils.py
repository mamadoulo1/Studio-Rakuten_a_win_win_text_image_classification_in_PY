"""
module listing preprocessing functions of Rakuten projet
"""
# preprocessing image library
import cv2

# Web UI for demo
import streamlit as st

# Data Science useful python librairies
import numpy as np

# library for regular expresssion handling
import re

# preprocessing text libraries
import nltk

nltk.download("punkt")
nltk.download("stopwords")
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from langdetect import detect
from googletrans import Translator

translator = Translator(service_urls=["translate.google.com"])
import spacy

# large pretrained pipeline model
nlp = spacy.load("fr_core_news_lg")


def delete_html_tags(text):
    """
    input  : text perhalps extracted from web page by web scraping
    return : text without html tags
    """
    text_without_tags = BeautifulSoup(text, "lxml").get_text(separator=" ")
    # to remove multiple spaces.
    text_without_tags = re.sub(r"  +", " ", text_without_tags)
    return text_without_tags


def translate_text(text):
    """
    language translation to french
    input  : text in any language
    return : text in french returned by google translate service
    """
    lang_found = detect(text)
    if lang_found == "fr":
        french_text = text
        # print("french language")
    else:
        try:
            # print("detected language :", lang_found)
            if len(text) > 5000:
                # print("text cut to googletranslate 5000 characters limit")
                french_text = translator.translate(
                    text[:5000], src=lang_found, dest="fr"
                ).text
            else:
                # print("length of text under googletranslate limit so not truncated")
                french_text = translator.translate(text, src=lang_found, dest="fr").text
        except:
            french_text = text
            st.error("PB with Google translate , text kept in its foreign language")

    return french_text


def prepare_text(text):
    """
    This function put in lowercase, suppress numbers and accents
    input  : raw text in french
    return : text in lowercase , without numbers and accents
    """

    # set in lowercase as stopwords to apply in next step are expressed in lowercase
    preptext = text.lower()
    # remove numbers
    preptext = re.sub(r"[0-9]+", "", preptext)
    # remove accents
    preptext = re.sub(r"é", "e", preptext)
    preptext = re.sub(r"è", "e", preptext)
    preptext = re.sub(r"î", "i", preptext)
    preptext = re.sub(r"â", "a", preptext)
    preptext = re.sub(r"ô", "o", preptext)
    preptext = re.sub(r"ë", "e", preptext)
    preptext = re.sub(r"ê", "e", preptext)
    preptext = re.sub(r"à", "a", preptext)
    preptext = re.sub(r"ã", "a", preptext)
    preptext = re.sub(r"û", "u", preptext)
    preptext = re.sub(r"¿", "", preptext)
    preptext = re.sub(r"\.", " ", preptext)
    # replace hyphens with spaces
    preptext = re.sub(r"-", " ", preptext)

    return preptext


def build_token_from_text(text):
    """
    This function converts text into a list of word tokens
    input  : text already prepared
    return : tokens from text
    """
    return word_tokenize(text)


def apply_stopwords(list_tokens):
    """
    This function suppress french stopwords ,
    nltk default list updated with additional items applied on dataset for algo training
    input : list of tokens
    return : reduced list of tokens without stopwords list AND token less than 3 characters filtered
    """
    fr_stopwords = set(stopwords.words("french"))
    fr_stopwords.update(
        [
            ":",
            "/",
            "//",
            "(",
            ")",
            "N°",
            "n°",
            "%",
            "?",
            "+",
            ".",
            "&",
            "[",
            "]",
            "*",
            "''",
            "``",
            "'",
            "////",
            "br/",
        ]
    )
    fr_stopwords.update(
        [
            "°",
            "@",
            "xcm",
            "-cm",
            "kg/m",
            "kg/",
            "g/m",
            "m³/h",
            "m²",
            "gr/m²",
            "_",
            "#",
            ";",
            "ø",
            "--",
            "²",
            "_-",
            "s'en",
            "s'il",
            "n'est",
            "d'un",
            "d'une",
            "...",
            "g/m²",
            "m/h",
            "c'est",
            "qu'il",
            "qu'elle",
            "jusqu",
        ]
    )

    list_stopwords_colours = [
        "rose",
        "gris",
        "blanc",
        "noir",
        "vert",
        "bleu",
        "rouge",
        "marron",
        "black",
        "anthracite",
        "jaune",
    ]
    list_stopwords_adjectives = [
        "bon",
        "nouveau",
        "facile",
        "rare",
        "commune",
        "grand",
        "petit",
        "mini",
        "deux",
        "ronde",
        "bas",
        "chaud",
        "pliable",
        "neuf",
        "ovale",
    ]
    list_stopwords_adjectivesbis = [
        "doux",
        "confortable",
        "new",
        "nouvelle",
        "nouveaux",
        "grande",
        "complet",
        "integrale",
        "legere",
    ]
    list_stopwords_adverb = [
        "afin",
        "enfin",
        "chaque",
        "plus",
        "contre",
        "dont",
        "sans",
        "pcs",
        "comment",
        "plusieurs",
        "sous",
        "chez",
        "entre",
        "tous",
        "depuis",
        "aussi",
        "alors",
    ]
    list_stopwords_adverbis = [
        "egalement",
        "tout",
        "toutes",
        "comme",
        "ainsi",
        "assez",
        "jamais",
        "encore",
        "lorsque",
        "tres",
        "toujours",
        "apres",
        "quand",
        "grace",
    ]
    list_stopwords_adverbter = [
        "particulierement",
        "beaucoup",
        "suffisamment",
        "pendant",
        "certainement",
        "immediatement",
        "seulement",
        "doucement",
        "neanmoins",
        "meme",
        "avant",
        "environ",
        "peu",
        "propos",
        "legerement",
        "aujourd'hui",
        "trop",
        "souvent",
    ]
    list_stopwords_various_names = [
        "necessaire",
        "facilement",
        "non",
        "peut",
        "ans",
        "mois",
        "attention",
    ]
    list_stopwords_various_namesbis = [
        "description",
        "comprend",
        "qualite",
        "haute",
        "merci",
        "ci-dessus",
        "caracteristiques",
        "couleur",
        "couleurs",
    ]
    list_stopwords_various_namester = [
        "annee",
        "annees",
        "taille",
        "inclus",
        "dimensions",
        "poids",
        "difference",
        "different",
    ]
    list_stopwords_verbs = [
        "etre",
        "avoir",
        "donnons",
        "vendons",
        "reversons",
        "peuvent",
        "pouvez",
        "gardez",
        "permettre",
        "n'existe",
        "utiliser",
        "utilise",
        "fait",
        "plait",
        "parlait",
    ]
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

    tokens_filtered = [mot for mot in list_tokens if mot not in fr_stopwords]
    tokens_filtered_min_3characters = []
    for i in range(len(tokens_filtered)):
        if len(tokens_filtered[i]) >= 3:
            tokens_filtered_min_3characters.append(tokens_filtered[i])
    return tokens_filtered_min_3characters


def lemmatization_token(token_list):
    """
    This function applies the lemmatization on received tokens
    input  : list of tokens to be retransformed artificially into a sentence internally
    return : list of lemmatized tokens AND token less than 3 characters filtered
    """
    str_tokens = ""
    str_tokens += " ".join(token_list)

    tokenized_str = nlp(str_tokens)
    new_tokens_after_lemmatization = []
    for token in tokenized_str:
        new_tokens_after_lemmatization.append(token.lemma_)

    # to clean side effect of spacy lemmatization
    # suppress tokens less than 3 characters
    tokens_kept = []
    for i in range(len(new_tokens_after_lemmatization)):
        if len(new_tokens_after_lemmatization[i]) >= 3:
            tokens_kept.append(new_tokens_after_lemmatization[i])

    return tokens_kept


def create_str_tokens(token_list):
    """This function builds a string from a list of received tokens
    input  : a list of final tokens
    return : a string of tokens separated with a space
    """
    str_tokens = ""
    str_tokens += " ".join(token_list)

    re.sub(r",", "", str_tokens)
    re.sub(r"'", "", str_tokens)
    re.sub(r"\[", "", str_tokens)
    re.sub(r"\]", "", str_tokens)

    return str_tokens


def preprocessing_text(text):
    """
    this function handles the following items :
    clean the HTML tags, translate if necessary,
    put in lowercase, suppress numbers and accents,
    build tokens list from text, suppress tokens considered stopwords in french,
    apply a lemmatization and finally create a string of final kept tokens

    input  : text in any language
    return : processed text as string of tokens equal to designation_str_tokens column of CSV used for algo training

    """
    text_without_tags = delete_html_tags(text)
    french_text = translate_text(text_without_tags)
    prep_text = prepare_text(french_text)
    tokens_from_text = build_token_from_text(prep_text)
    final_tokens = apply_stopwords(tokens_from_text)
    lemmatized_tokens = lemmatization_token(final_tokens)
    str_tokens = create_str_tokens(lemmatized_tokens)
    return str_tokens


def suppress_white_areas(path, name_img):
    """this function crops a image suppressing surrounding white areas
    inputs : path where is located the image to process, path where to store the new image , image name
    return : no
    """
    full_name = path + name_img
    name_img_after_preprocessing = "cropped_" + name_img
    img_color = cv2.imread(full_name, cv2.IMREAD_COLOR)  # read the image
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    gray = 255 * (gray < 192).astype(np.uint8)  # To invert the text to white
    # often threshold 128 is used but objects present in some pictures are partially damaged so 192 suits better :-)
    coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    rect = img_color[
        y : y + h, x : x + w
    ]  # Crop the image - note we do this on the original image
    # !!!!!! important choice to keep or not original images !!!!
    # cv2.imwrite(path_complet, rect)# save the image (original image lost)
    #
    full_name_new = path + name_img_after_preprocessing
    cv2.imwrite(
        full_name_new, rect
    )  # save the image in another directory (keeping original image)
    return name_img_after_preprocessing
