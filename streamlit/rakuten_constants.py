"""
constants of Rakuten project
"""
#target classes of the project
#
num_classes = 27
#class names given by our team
name_class = ['10 - editions','2280 - magazines',"50 - equipement_playstation","1280 - jouets_enfant","2705 - livres","2522 - fournitures_scolaires","2582 - materiel_jardin","1560 - mobilier_interieur","1281 - jeux_enfants","1920 - literie","2403 - livres_par_lot","1140 - figurine","2583 - materiel_piscine","1180 - univers_de_fiction","1300 - equipements_telecommandes","2462 - materiel_jeux_video","1160 - jeux_de_cartes_collection","2060 - decoration","40 - jeux_video","60 - console_de_jeu","1320 - materiel_bebe","1302 - peche_lampe_plein_air","2220 - materiel_animaux_compagnie","2905 - jeux_sur_PC","2585 - accessoires_outil","1940 - articles_nourriture","1301 - vetements_nouveau-né_et_billard_flechettes"] 
#Rakuten classes
list_class = [10, 2280 ,  50 ,1280 ,2705, 2522, 2582, 1560, 1281, 1920, 2403, 1140, 2583, 1180, 1300 ,2462, 1160, 2060 , 40,   60 ,1320 ,1302 ,2220 ,2905, 2585, 1940 ,1301]       
#encoded classes used by Deep Learning network
list_class_onehot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
#image shape given to VGG16 pretrained model
target_size = (150,150,3)
#max words composing text given to embedding layer 
text_size=200
#variable used by tokenizer (value determined by a specific run of tokenizer)and kept equal for embedding layer then
max_features = 47900