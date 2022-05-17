import matplotlib.pyplot as plt
import streamlit as st
import sys, os
import pickle
import cv2

from PIL import Image
from wordcloud import WordCloud
import requests
import numpy as np


from tensorflow.keras.models import load_model

# import Rakuten project utils functions
from rakuten_constants import *
from rakuten_preprocessing_utils import *
from rakuten_processing_utils import *

# file used to save the model after each epoch if best weights on gDrive
path_name_model_best_weights = "C:\\Users\\christophe\\Documents\\projet_Rakuten\\Rakuten_multimodal.weights.270122.hdf5"

DEST = "C:\\Users\\christophe\\Documents\\projet_Rakuten\\predict\\images\\"

list_images_5samples = [
        "image_row0.jpg",
        "image_row1.jpg",
        "image_row2.jpg",
        "image_row3.jpg",
        "image_row4.jpg"
]


text_row0 = "Olivia: Personalisiertes Notizbuch / 150 Seiten / Punktraster / Ca Din A5 / Rosen-Design"
text_row1 = "Journal Des Arts (Le) N° 133 Du 28/09/2001 - L'art Et Son Marche Salon D'art Asiatique A Paris - Jacques Barrere - Francois Perrier - La Reforme Des Ventes Aux Encheres Publiques - Le Sna Fete Ses Cent Ans."
text_row2 = "Grand Stylet Ergonomique Bleu Gamepad Nintendo Wii U - Speedlink Pilot Style PILOT STYLE Touch Pen de marque Speedlink est 1 stylet ergonomique pour GamePad Nintendo Wii U. Pour un confort optimal et une précision maximale sur le GamePad de la Wii U: ce grand stylet hautement ergonomique est non seulement parfaitement adapté à votre main mais aussi très élégant. Il est livré avec un support qui se fixe sans adhésif à l'arrière du GamePad Caractéristiques: Modèle: Speedlink PILOT STYLE Touch Pen Couleur: Bleu Ref. Fabricant: SL-3468-BE Compatibilité: GamePad Nintendo Wii U Forme particulièrement ergonomique excellente tenue en main Pointe à revêtement longue durée conçue pour ne pas abîmer l'écran tactile En bonus : Support inclu pour GamePad "
text_row3 = "Peluche Donald - Europe - Disneyland 2000 (Marionnette À Doigt) "
text_row4 = "La Guerre Des Tuques Luc a des idées de grandeur. Il veut organiser un jeu de guerre de boules de neige et s'arranger pour en être le vainqueur incontesté. Mais Sophie s'en mêle et chambarde tous ses plans..."
list_text_5samples = [text_row0, text_row1, text_row2, text_row3, text_row4]

true_label_5samples = [0, 1, 2, 3, 4]

list_URL_10img_site_Rakuten = [
        "https://fr.shopping.rakuten.com/photo/nintendo-switch-oled-blanche-1860235496_ML.jpg",
        "https://fr.shopping.rakuten.com/photo/1810590119_L_NOPAD.jpg",
        "https://fr.shopping.rakuten.com/photo/1447893247_L_NOPAD.jpg",
        "https://fr.shopping.rakuten.com/photo/1271650237_L_NOPAD.jpg",
        "https://fr.shopping.rakuten.com/photo/964631061_L_NOPAD.jpg",
        "https://fr.shopping.rakuten.com/photo/matelas-160x200-a-memoire-de-forme-ergo-therapy-1077590700_ML.jpg",
        "https://fr.shopping.rakuten.com/photo/1225683711_L_NOPAD.jpg",
        "https://fr.shopping.rakuten.com/photo/1887024517_L_NOPAD.jpg",
        "https://fr.shopping.rakuten.com/photo/1960752072_L_NOPAD.jpg",
        "https://fr.shopping.rakuten.com/photo/1905383763_L_NOPAD.jpg",
        "https://fr.shopping.rakuten.com/photo/1810590119_L_NOPAD.jpg",
        "https://fr.shopping.rakuten.com/photo/1960752072_L_NOPAD.jpg"
]

list_10img_site_Rakuten = [
        "console.jpg",
        "livre.jpg",
        "piscine.jpg",
        "jouet_enfant.jpg",
        "cahier.jpg",
        "matelas.jpg",
        "carte_jeu_collection.jpg",
        "chaussons.jpg",
        "rideau.jpg",
        "mobilier_interieur.jpg",
        "livrebis.jpg",
        "rideaubis.jpg"
]

#executed once only to store locally the 10 images downloaded from Rakuten
#images stored now under github project 
#for i in range(10):
#    response = requests.get(list_URL_10img_site_Rakuten[i], stream=True)
#    pil_image = Image.open(response.raw)
#    open_cv_image = np.array(pil_image) 
#    full_name = DEST + list_10img_site_Rakuten[i]
#    cv2.imwrite(full_name,open_cv_image) 
    

text1 = "Nintendo Switch Oled - Blanc"
text2 = "Harry Potter Tome 5 - Harry Potter Et L'ordre Du Phénix"
text3 = "Intex Piscine design Metal Frame 366 x 76 cm bleu"
text4 = "Pokémon Pack De 8 Figurines"
text5 = "Cahier Spirale A4 100 Pages 70g Quadrillé 5x5"
text6 = "Matelas 160x200 À Mémoire De Forme Ergo-Therapy"
text7 = "Lot De 50 Cartes Dragon Ball Super + Cartes Brillantes En Cadeaux"
text8 = "Chaussons Bébé Licence Fantaisie Pack De 4 Chaussons Mickey And Co"
text9 = "Rideau De Fenêtre En Voile Panneau De Rideau Décoration Maison ...."
text10 = "Ensemble Noir Table 2 Chaises Cuisine Salle À Manger Repas Cadre Tubes Acier Accessoire Mobilier Meuble Maison Intérieur"
text11 = "Harry Potter Tome 5 - Harry Potter Et L'ordre Du Phénix. A quinze ans, Harry s'apprête à entrer en cinquième année à Poudlard. Et s'il est heureux de retrouver le monde des sorciers, il n'a jamais été aussi anxieux. L'adolescence, la perspective des examens importants en fin d'année et ces étranges cauchemars... Car Celui-Dont-On-Ne-Doit-Pas-Prononcer-Le-Nom est de retour et, plus que jamais, Harry sent peser sur lui une terrible menace. Une menace que le ministère de la Magie ne semble pas prendre au sérieux, contrairement à Dumbledore. Poudlard devient alors le terrain d'une véritable lutte de pouvoir. La résistance s'organise autour de Harry qui va devoir compter sur le courage et la fidélité de ses amis de toujours... D'une inventivité et d'une virtuosité rares, découvrez le cinquième tome de cette saga que son auteur a su hisser au rang de véritable phénomène littéraire."
text12 = "Rideau De Fenêtre En Voile Panneau De Rideau Décoration Maison .... Ajout de style à une fenêtre ou une porte, ce qui rend votre maison extrêmement fascinant et charmant, créant une atmosphère confortable et romantique pour votre famille. Peut être utilisé comme un rideau de porte, fenêtre rideau ou un arrière-plan pour un affichage de vitrine. Ce rideau de porte peut être utilisé comme cloisons entre les chambres ou l'utilisation comme un écran pour empêcher les mouches. Peut être utilisé dans la maison, hôtel, café, bureau et autres lieux."
list_10text_site_Rakuten = [text1, text2, text3, text4, text5,text6,text7,text8,text9,text10,text11,text12]

#@st.cache
# While caching the return value of chargement_model(), 
# Streamlit encountered an object of type keras.engine.functional.Functional,
# which it does not know how to hash.
#@st.experimental_memo ne fonctionne pas non nul
@st.experimental_singleton
def chargement_model():
    loaded_model = load_model(path_name_model_best_weights)
    return loaded_model

@st.cache
def chargement_tokenizer():
    # load tokenizer used when model was trained
    with open(
        "C:\\Users\\christophe\\Documents\\projet_Rakuten\\Rakuten_tokenizer.pickle", "rb"
    ) as handle:
        loaded_tokenizer = pickle.load(handle)
    return loaded_tokenizer

#@st.cache ne fonctionne pas
@st.experimental_singleton(suppress_st_warning=True)
def apply_prediction_from_ds(DEST,text,image,target_label):

    test_pred_class = []
    three_best_classes = []
    y_test_class = []

    cropped_image_name = suppress_white_areas(DEST, image)
    cropped_image_name = DEST + cropped_image_name
    img_tensor = transform_image(cropped_image_name, target_size)

    str_tokens = preprocessing_text(text)
    txt_tensor = transform_text(str_tokens, text_size, tokenizer)
    
    test_pred = multimodal_sav_demo.predict([img_tensor, txt_tensor])
    test_pred_class_img = test_pred.argmax(axis=1)
    # this highest probability set to 0 to find out the second better
    test_pred[0][test_pred_class_img] = 0
    test_pred_class_img2 = test_pred.argmax(axis=1)
    # this second probability set to 0 to find out the thrid better
    test_pred[0][test_pred_class_img2] = 0
    test_pred_class_img3 = test_pred.argmax(axis=1)
    test_pred_class.append(test_pred_class_img[0])
    # we keep 3 classes with best probabilities given by softmax at output of the model
    three_best_classes.append(test_pred_class_img[0])
    three_best_classes.append(test_pred_class_img2[0])
    three_best_classes.append(test_pred_class_img3[0])

    y_test_class.append(target_label)

    st.write(
        "classe Rakuten  : {0}   et     celle prédite  : {1}".format(
                name_class[y_test_class[0]], name_class[test_pred_class[0]]
        )
    )
    st.write("classes potentielles suivantes selon les probabilités décroissantes de la fct softmax :")
    st.write("{0} et {1}".format(name_class[three_best_classes[1]],
                name_class[three_best_classes[2]])
    )

    st.write("texte associé : ")
    st.write(text)
    st.write("et l'image associée : ")
    full_name = DEST + image
    img_color = cv2.imread(full_name, cv2.IMREAD_COLOR)  # read the image
    fig_img = plt.figure(figsize=(20, 10))
    plt.imshow(img_color)
    st.pyplot(fig_img)

#@st.cache ne fonctionne pas
@st.experimental_singleton(suppress_st_warning=True)
def apply_prediction_from_Rakuten(DEST,text,image):

    test_pred_class_r = []
    three_best_classes_r = []

    cropped_image_name = suppress_white_areas(DEST, image)
    cropped_image_name = DEST + cropped_image_name
    img_tensor = transform_image(cropped_image_name, target_size)

    str_tokens = preprocessing_text(text)
    txt_tensor = transform_text(str_tokens, text_size, tokenizer)

    test_pred_r = multimodal_sav_demo.predict([img_tensor, txt_tensor])
    test_pred_class_img_r = test_pred_r.argmax(axis=1)
    # this highest probability set to 0 to find out the second better
    test_pred_r[0][test_pred_class_img_r] = 0
    test_pred_class_img2_r = test_pred_r.argmax(axis=1)
    # this second probability set to 0 to find out the thrid better
    test_pred_r[0][test_pred_class_img2_r] = 0
    test_pred_class_img3_r = test_pred_r.argmax(axis=1)
    test_pred_class_r.append(test_pred_class_img_r[0])
    # we keep 3 classes with best probabilities given by softmax at output of the model
    three_best_classes_r.append(test_pred_class_img_r[0])
    three_best_classes_r.append(test_pred_class_img2_r[0])
    three_best_classes_r.append(test_pred_class_img3_r[0])

    st.write(
        "classe prédite  : {0}".format(
                name_class[test_pred_class_r[0]]
        )
    )
    st.write("classes potentielles suivantes selon les probabilités décroissantes de la fct softmax :")
    st.write("{0} et {1}".format(name_class[three_best_classes_r[1]],
                name_class[three_best_classes_r[2]])
    )

    st.write("texte associé : ")
    st.write(text)
    st.write("et l'image associée : ")
    full_name = DEST + image
    img_color = cv2.imread(full_name, cv2.IMREAD_COLOR)  # read the image
    fig_img = plt.figure(figsize=(20, 10))
    plt.imshow(img_color)
    st.pyplot(fig_img)




multimodal_sav_demo = chargement_model()
tokenizer = chargement_tokenizer()

# gestion des différents paths 
def imgpath(script_dir, img):
    return script_dir + os.path.normpath('\\' + img)

def write():
    """Used to write the page in the app.py file"""

    st.title("Prédiction de la classe d'un produit basée sur son texte et son image")

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # In honour of our final customer Rakuten, japanese company
    mask_Rakuten = np.array(Image.open(imgpath(SCRIPT_DIR,"il_canale_grande.png")))
    #In order to correct use of the mask transform 0 to 255
    mask_Rakuten[mask_Rakuten == 0] = 255

    # words cloud object configuration for generating and drawing
    # size of words cloud (27 classes)
    size_top = 27
    stop_words = []
    wc = WordCloud(background_color="white", max_words=size_top, stopwords=stop_words, max_font_size=20, random_state=42)
    wc_Rakuten = WordCloud(background_color="black", max_words=size_top, stopwords=stop_words, mask = mask_Rakuten, max_font_size=20, random_state=42)

    # build a string with all classes list
    name_class_wc = ['editions','magazines',"equipement_playstation","jouets_enfant","livres","fournitures_scolaires","materiel_jardin","mobilier_interieur","jeux_enfants","literie","livres_par_lot","figurine","materiel_piscine","univers_de_fiction","equipements_telecommandes","materiel_jeux_video","jeux_de_cartes_collection","decoration","jeux_video","console_de_jeu","materiel_bebe","peche_lampe_plein_air","materiel_animaux_compagnie","jeux_sur_PC","accessoires_outil","articles_nourriture","vetements_nouveau_né_et_billard_flechettes"] 
    text = ' '.join(name_class_wc)

    wc_Rakuten.generate(text)
    fig_japan = plt.figure(figsize=(3, 3))
    plt.imshow(wc_Rakuten)
    plt.axis("off")
    plt.title('Les 27 classes du challenge Rakuten')

    st.pyplot(fig_japan)

    st.subheader("Prédiction sur 5 produits de la database originelle")
    
    liste_articles = ["article_1","article_2","article_3","article_4","article_5"]
    valeur_du_select = st.selectbox('Select', liste_articles)
    indice = 0
    if (valeur_du_select == "article_1"):
        indice = 0
    elif (valeur_du_select == "article_2"):
        indice = 1
    elif (valeur_du_select == "article_3"):
        indice = 2
    elif (valeur_du_select == "article_4"):
        indice = 3
    elif (valeur_du_select == "article_5"):
        indice = 4

    _ = apply_prediction_from_ds(DEST,list_text_5samples[indice],list_images_5samples[indice],true_label_5samples[indice])
    
    if st.button("vider le cache pour prédire 2 fois le meme article"):
        # Clear memorized values:
        apply_prediction_from_ds.clear()

    st.subheader("Prédiction sur 10 produits pris sur le site officiel Rakuten")
    st.write("[URL du site Rakuten France](https://fr.shopping.rakuten.com/)")

    liste_articles_Rakuten = ["console","livre","piscine","jouet_enfant","cahier","matelas","carte_jeu_collection","chaussons","rideau_decoration_maison","mobilier_interieur","livre_bis","rideau_decoration_maison_bis"]
    valeur_du_select = st.selectbox('Select', liste_articles_Rakuten)
    ind = 0
    if (valeur_du_select == "console"):
        ind = 0
    elif (valeur_du_select == "livre"):
        ind = 1
    elif (valeur_du_select == "piscine"):
        ind = 2
    elif (valeur_du_select == "jouet_enfant"):
        ind = 3
    elif (valeur_du_select == "cahier"):
        ind = 4
    elif (valeur_du_select == "matelas"):
        ind = 5
    elif (valeur_du_select == "carte_jeu_collection"):
        ind = 6
    elif (valeur_du_select == "chaussons"):
        ind = 7
    elif (valeur_du_select == "rideau_decoration_maison"):
        ind = 8
    elif (valeur_du_select == "mobilier_interieur"):
        ind = 9
    elif (valeur_du_select == "livre_bis"):
        ind = 10
    elif (valeur_du_select == "rideau_decoration_maison_bis"):
        ind = 11

    _ = apply_prediction_from_Rakuten(DEST,list_10text_site_Rakuten[ind],list_10img_site_Rakuten[ind])

    if st.button("vider le cache pour prédire 2 fois le meme produit"):
        # Clear memorized values:
        apply_prediction_from_Rakuten.clear()