# Projet Rakuten_a_win_win_text_image_classification_in_PY

# Contexte
Ce projet a été réalisé dans le cadre de la formation Data Scientist de Datascientest (promotion formation continue initiée en mai 2021) 
et du challenge Rakuten France Multimodal Product Data Classification.

# Introduction

Ce dossier couvre les phases d’activités que notre groupe a mené sur le projet fil rouge de la formation de data scientist.
Ainsi, il est composé de l’étude préliminaire des données texte et image à exploiter, suivi de leur prétraitement.

Ce projet s’inscrivant dans le domaine du NLP qui fourmille d’innovations nous a permis de tester de nombreuses techniques particulièrement sur les méthodes d’embedding , extraction de features texte sous forme de vecteurs numériques dense. 

L’approche méthodologique décomposée en une première phase baseline (réponse court terme très utile en début de projet) fut basée sur les algorithmes de Machine Learning de type différents (LR, LGB et RF) pour avoir une base intéressante de comparaison. 

Ensuite, les expérimentations itératives, nous ont amené à développer d’une part des modèles de Deep Learning sur le framework tensorflow/keras de type séquentiel servant aux analyses unimodales puis fonctionnel pour bâtir une étude multimodale et d’autre part à utiliser la puissance de transfert learning ou d’un transformateur encodeur pour générer des vecteurs embedding sur le texte , features qui furent ensuite passées à un modèle ML de classification.


# Objectif 
Le problème consiste, à partir d'informations de textes et d'une image associés à chaque produit du catalogue, à classifier automatiquement ceux ci avec la meme taxinomie que Rakuten avec le moins d'erreurs possibles.<br />
En réalité, la taxinomie Rakuten comprend plus de 1000 catégories, mais dans le cadre de ce challenge, l'étude proposée est limitée à seulement 27 de celles ci.

# Jeu de données
Lors de l'inscription au challenge , Rakuten met à disposition un jeu de données pour l'entrainement (et la validation) de 84916 observations 
et un jeu de test de 13812 articles sans label pour pouvoir déposer une réponse de classification associée au concours.

Chaque échantillon représente un article du catalogue de la plateforme de e-commerce Rakuten France, et comporte nécessairement un champ textuel désignation ainsi qu'une image , et potentiellement une description textuelle plus détaillée additionnelle.

# Description des fichiers
doc : dossier qui contient les rapports publiés dans le cadre de ce projet<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;incluant Rapport_final_Projet1_Rakuten161221_version3.docx<br />

streamlit : dossier qui contient les élements utilisés par ce framework démonstrateur web de nos applications ML et DL pour la soutenance projet<br />
**le script python Rakuten_text_image_classification_App.py intégré à streamlit correspond au développement du modéle multimodal DL**<br />
c'est à dire entrainement , évaluation et exemples de prédiction sur les 5 premiers produits du dataset originel<br />

EDA_DataViz_Preprocessing : dossier qui contient les notebooks d'audit des données, certaines visualisations et la traduction des textes en francais<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Rakuten_project1_steps_eda_dataviz_070921.ipynb : première exploration des données<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Rakuten_project_steps_eda_image_080921.ipynb : visualisation des images et suppression des bords blancs (crop en anglais)

**2 modéles de classification choisis pour la généralisation :**<br />
le premier utilisant un algorithme ML exploitant le texte<br />
et le second basé sur une architecture DL multimodale (texte + image)<br />

**rakuten_text_version_finale.ipynb** : classification du texte (ML SVM)
**gColab_Rakuten_project_final_processing_multimodal_161221.ipynb** : classification bimodale avec texte et image<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;( (embedding + RNN GRU) + VGG16 sur le framework tensorflow)<br />



Modélisations_expérimentales : dossier qui contient les notebooks de modélisations livrés à chaque étape de sprint projet<br />
Note : ces notebooks sont également intéressants car ils traduisent nos explorations/études de techniques trés interessantes de NLP texte


# Membres de l’équipe projet

Haeji YUN<br />
Mamadou LO [LinkedIn](https://www.linkedin.com/in/mamadou-lo-1047361b9/)<br />
Christophe Paquet [LinkedIn](https://www.linkedin.com/in/c-paquet-machine-and-deep-learning-for-fun)<br />

Et Emilie Greff , notre mentor !
