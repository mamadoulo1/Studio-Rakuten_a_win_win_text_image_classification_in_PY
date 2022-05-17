"""
main module of Rakuten project
"""
# System
import sys
import os
import time

# various tools libraries
import re
import itertools
from pathlib import Path
import pickle
from math import ceil

# preprocessing image library
import cv2

import warnings

warnings.filterwarnings("ignore")

# visualisation library
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

# Web UI for demo
import streamlit as st

# Data Science useful python librairies
import numpy as np
import pandas as pd

# DL framework
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from livelossplot import PlotLossesKerasTF

from tensorflow.keras.layers import Dense, Dropout, Flatten, Embedding, GRU
from tensorflow.keras.layers import BatchNormalization, Concatenate, Activation
from tensorflow.keras import activations
from tensorflow.keras.applications.vgg16 import VGG16

# scikit learn library
from sklearn.model_selection import train_test_split
from sklearn import metrics

# various tools libraries
from IPython.display import display


# import Rakuten project utils functions
from rakuten_constants import *
from rakuten_preprocessing_utils import *
from rakuten_processing_utils import *

GPU, CPU, BESTDEV = None, None, None


def set_best_device():
    """function that determines if GPU is available
    otherwise BESTDEV is set to CPU
    return  : GPU , CPU and BESTDEV devices
    """
    devices = tf.config.list_physical_devices()
    # beginning /physical_device  is subtracted
    for dev in devices:
        if dev.device_type == "GPU":
            tokens = dev.name.split(":")[-2:]
            name = ":".join(tokens)
            GPU = "/device:" + name
            st.write(f"GPU device found: '{GPU}'")
            BESTDEV = GPU
        elif dev.device_type == "CPU":
            name = dev.name
            tokens = dev.name.split(":")[-2:]
            name = ":".join(tokens)
            CPU = "/device:" + name
            st.write(f"CPU device found: '{CPU}'")
    if not GPU:
        st.text("GPU device NOT found")
        BESTDEV = CPU
    st.write(f"Best device is: '{BESTDEV}'")
    return GPU, CPU, BESTDEV


# Operations that rely on a random seed actually derive it from two seeds: the global and operation-level seeds.
# This sets the global seed.
tf.random.set_seed(42)

# If both the global and the operation seed are set:
# Both seeds are used in conjunction to determine the random sequence.

# file used to save the model after each epoch if best weights on gDrive
path_name_model_best_weights = "C:\\Users\\admin\\Documents\\projet_Rakuten\\Rakuten_multimodal.weights.270122.hdf5"

PATH_WORK = "C:\\Users\\admin\\Documents\\"

# go to working directory
os.chdir(PATH_WORK)


def load_dataset():
    """function that reads the CSV file generated from EDA-prerprocessing phases
    return : dataframe
    """
    # load dataset containing text of products and set the image path in this environment
    data_train = pd.read_csv("X_train_rakuten_afterEDA_preprocessing.csv", index_col=0)
    data_train = data_train.reset_index(drop=False)
    data_train["nom_img"] = (
        r"C:\\Users\\admin\\Documents\\projet_Rakuten\\cropped_image_train\\"
        + data_train["nom_img"]
    )
    st.write("dimension du dataset :", data_train.shape)

    # display dataset
    # 3 main columns :
    # product images (nom_img), preprocessed product text (designation_token_final)
    # and target class (prdtypecode)
    st.dataframe(data_train.head())

    return data_train


def display_relationship_encoded_classes():
    """function that displays relationship between encodes classes for DL model
    and classes originated from Rakuten challenge
    and class names given by team project during EDA
    """
    st.text("association between encoded classes passed to the DL model")
    st.text("and challenge Rakuten classes and related name choosen by our team")
    df_link_class_nbr_name = pd.DataFrame(
        {
            "encoded_class": list_class_onehot,
            "original class Rakuten": list_class,
            "project class name": name_class,
        }
    )

    st.dataframe(df_link_class_nbr_name.head(len(df_link_class_nbr_name)))


def text_length_explanation_for_model(data_train):
    """
    function that displays the histogram of number of words per text
    """
    # cell that explains the choosen value of text_size constant
    word_count = data_train.designation_str_tokens.apply(lambda x: len(x.split(" ")))
    fig = plt.figure(figsize=(15, 10))
    sns.distplot(word_count)
    plt.xlim(0, 300)
    plt.title("words count distribution")
    st.pyplot(fig)

    # choice to keep only 200 words per product text
    st.write(
        "98pct of texts describing products contain less than {} words explaining threshold of 200 used for text length".format(
            word_count.quantile(0.98)
        )
    )


def prepare_train_valid_test_df(data_train):
    """
    function that splits randomly in three sets the data :
    one for training purpose, one for validation after each epoch
    and finally the test one unknown by model to evaluate a future generalization
    return : dataframes trai, valmid and test
    """
    # split in training and validation sets for training phase and test set for model evaluation
    target = data_train.prdtypecode
    data_train = data_train.drop(
        [
            "designation_orig",
            "lang_designation",
            "designation",
            "designation_token_final",
            "prdtypecode",
        ],
        axis=1,
    )

    # Divide randomly data into training and (validation/test) sets
    # maintaining proportion of classes equal (stratify option)
    X_train, X_validtest, y_train, y_validtest = train_test_split(
        data_train, target, test_size=0.3, random_state=42
    )
    # Divide randomly data into validation and test set
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_validtest, y_validtest, test_size=0.33, random_state=42
    )

    # for training purpose
    df_train = pd.concat([X_train, y_train], axis=1)
    df_train = df_train.reset_index(drop=True)
    df_valid = pd.concat([X_valid, y_valid], axis=1)
    df_valid = df_valid.reset_index(drop=True)

    # for generalization purpose
    df_test = pd.concat([X_test, y_test], axis=1)
    df_test = df_test.reset_index(drop=True)

    # df_train and df_valid will be used to build the batchs !!!!!!
    df_train = create_col_class_onehot(df_train)
    df_valid = create_col_class_onehot(df_valid)

    # df_test will be used to evaluate the model
    df_test = create_col_class_onehot(df_test)

    st.text("")
    st.write("shape of training set    :", df_train.shape)
    st.write("shape of  validation set :", df_valid.shape)
    st.write("shape of test set        :", df_test.shape)

    return df_train, df_valid, df_test


def train_tokenizer(df_train):
    """
    function that creates and trains a tokenizer on train set corpus
    return : trained tokenizer
    """
    # tokenizer created and trained on specific corpus (limited to texts of train set)
    tokenizer = create_tokenizer(df_train["designation_str_tokens"])

    # saving
    st.text("")
    st.text(
        "trained tokenizer on train set corpus is saved on disk with name Rakuten_tokenizer.pickle"
    )
    with open(
        "C:\\Users\\admin\\Documents\\projet_Rakuten\\Rakuten_tokenizer.pickle", "wb"
    ) as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return tokenizer


def instanciate_model():
    """
    function that instanciates the multimodal DL neural network
    with both text and image inputs
    return : instanciated model
    """
    # instanciate a model with two input tensors and a classifier composed of 2 dense layers
    multimodal_model = multimodal_vgg16_emb_gru(target_size, text_size, max_features)
    # for notebook env , display inline but with streamlit ,image read on disk
    tf.keras.utils.plot_model(multimodal_model, show_shapes=True)
    st.image("./model.png")

    # CategoricalCrossentropy : Computes the crossentropy loss between the labels and predictions
    # CategoricalAccuracy :  Calculates how often predictions match one-hot labels.
    # learning_rate kept finally to the default value 0.001 after study
    opt = Adam(learning_rate=0.001)
    multimodal_model.compile(
        optimizer=opt, loss=["categorical_crossentropy"], metrics=["accuracy"]
    )
    return multimodal_model


def train_model(multimodal_model, tokenizer, df_train, df_valid):
    """
    function that trains the model and displays the loss and accuracy history
    best weights are saved on disk
    callbacks EarlyStopping on val_loss and ModelCheckpoint are are used to interact with the model
    """
    # callback to stop training phase when val_loss is no more decreasing
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, mode="min", restore_best_weights=True
    )

    # Saves TF/Keras model after each epoch in case of new best weights associated with a new min of val_loss
    checkpointer = ModelCheckpoint(
        filepath=path_name_model_best_weights, verbose=1, save_best_only=True
    )

    # training of the functional model multimodal (Image + text)
    traingen = CustomDataGen(
        df_train,
        tokenizer=tokenizer,
        x_col_img={"nom_img": "nom_img"},
        x_col_txt={"designation_str_tokens": "designation_str_tokens"},
        y_col={"class_onehot": "class_onehot"},
        batch_size=32,
        input_size=target_size,
        input_text_size=text_size,
    )

    valgen = CustomDataGen(
        df_valid,
        tokenizer=tokenizer,
        x_col_img={"nom_img": "nom_img"},
        x_col_txt={"designation_str_tokens": "designation_str_tokens"},
        y_col={"class_onehot": "class_onehot"},
        batch_size=32,
        input_size=target_size,
        input_text_size=text_size,
    )

    start = time.time()
    st.header("training of model is starting")
    with tf.device(BESTDEV):  # explicitly activates GPU if available
        history = multimodal_model.fit(
            traingen,
            validation_data=valgen,
            epochs=10,
            callbacks=[early_stopping, checkpointer],
        )
    end = time.time()
    elapsed = round((end - start) / 60)

    st.write(f"\nmodel training duration : {elapsed:.2f} mn")
    st.write("model weights saved in file :", path_name_model_best_weights)

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(acc) + 1)
    fig_acc = plt.figure(figsize=(8, 6))
    plt.plot(epochs, acc, "bo", label="training")
    plt.plot(epochs, val_acc, "b", label="validation")
    plt.title("accuracy during training and validation for multimodal DL")
    plt.legend()
    st.pyplot(fig_acc)

    fig_loss = plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss, "bo", label="training")
    plt.plot(epochs, val_loss, "b", label="validation")
    plt.title("loss during training and validation for multimodal DL")
    plt.legend()
    st.pyplot(fig_loss)


def evaluation_model(df_test):
    """
    function that displays classifcation report and confusion matrix
    based of test set and related predictions
    and also shows the most important mismatched classifications
    """
    # evaluate the model using the test dataset
    st.header("start of model evaluation")

    multimodal_sav = load_model(path_name_model_best_weights)

    # load tokenizer used when model was trained
    with open(
        "C:\\Users\\admin\\Documents\\projet_Rakuten\\Rakuten_tokenizer.pickle", "rb"
    ) as handle:
        tokenizer = pickle.load(handle)

    test_pred_class = []
    y_test_class = []

    for i in range(len(df_test)):
        img_tensor = transform_image(df_test["nom_img"].iloc[i], target_size)
        txt_tensor = transform_text(
            df_test["designation_str_tokens"].iloc[i], text_size, tokenizer
        )
        test_pred = multimodal_sav.predict([img_tensor, txt_tensor])
        test_pred_class_img = test_pred.argmax(axis=1)
        test_pred_class.append(test_pred_class_img[0])

        y_test_class.append(df_test["class_onehot"].iloc[i])

    st.write(
        metrics.classification_report(
            y_test_class, test_pred_class, target_names=name_class
        )
    )

    cnf_matrix = metrics.confusion_matrix(y_test_class, test_pred_class)

    classes = range(0, 27)

    fig_cf = plt.figure(figsize=(15, 15))

    plt.imshow(cnf_matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix for multimodal DL model")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, list_class, rotation="vertical")
    plt.yticks(tick_marks, list_class)

    for i, j in itertools.product(
        range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])
    ):
        plt.text(
            j,
            i,
            cnf_matrix[i, j],
            horizontalalignment="center",
            color="white" if cnf_matrix[i, j] > (cnf_matrix.max() / 2) else "black",
        )

    plt.ylabel("True labels")
    plt.xlabel("Predicted Labels")
    st.pyplot(fig_cf)

    st.write(
        "mismatched classification listed when representing at least 10% of the right predictions amount of the class\n"
    )

    for i, j in itertools.product(
        range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])
    ):
        ten_per_cent = ceil(cnf_matrix[i, i] / 10)
        if cnf_matrix[i, j] > ten_per_cent and i != j:
            st.write(
                "class {0} ({4}) wrongly predicted to class {1} ({5}) , {2} for {3} correct predictions ".format(
                    list_class[i],
                    list_class[j],
                    cnf_matrix[i, j],
                    cnf_matrix[i, i],
                    name_class[i],
                    name_class[j],
                )
            )


def predict_model(tokenizer):
    """
    function that produces predictions of five first products of the original dataset
    """
    st.text("")
    st.header("start of model prediction demo")
    DEST = "C:\\Users\\admin\\Documents\\projet_Rakuten\\predict\\images\\"
    multimodal_sav_demo = load_model(path_name_model_best_weights)

    list_images_5samples = [
        "image_row0.jpg",
        "image_row1.jpg",
        "image_row2.jpg",
        "image_row3.jpg",
        "image_row4.jpg",
    ]
    text_row0 = "Olivia: Personalisiertes Notizbuch / 150 Seiten / Punktraster / Ca Din A5 / Rosen-Design"
    text_row1 = "Journal Des Arts (Le) N° 133 Du 28/09/2001 - L'art Et Son Marche Salon D'art Asiatique A Paris - Jacques Barrere - Francois Perrier - La Reforme Des Ventes Aux Encheres Publiques - Le Sna Fete Ses Cent Ans."
    text_row2 = "Grand Stylet Ergonomique Bleu Gamepad Nintendo Wii U - Speedlink Pilot Style PILOT STYLE Touch Pen de marque Speedlink est 1 stylet ergonomique pour GamePad Nintendo Wii U. Pour un confort optimal et une précision maximale sur le GamePad de la Wii U: ce grand stylet hautement ergonomique est non seulement parfaitement adapté à votre main mais aussi très élégant. Il est livré avec un support qui se fixe sans adhésif à l'arrière du GamePad Caractéristiques: Modèle: Speedlink PILOT STYLE Touch Pen Couleur: Bleu Ref. Fabricant: SL-3468-BE Compatibilité: GamePad Nintendo Wii U Forme particulièrement ergonomique excellente tenue en main Pointe à revêtement longue durée conçue pour ne pas abîmer l'écran tactile En bonus : Support inclu pour GamePad "
    text_row3 = "Peluche Donald - Europe - Disneyland 2000 (Marionnette À Doigt) "
    text_row4 = "La Guerre Des Tuques Luc a des idées de grandeur. Il veut organiser un jeu de guerre de boules de neige et s'arranger pour en être le vainqueur incontesté. Mais Sophie s'en mêle et chambarde tous ses plans..."
    list_text_5samples = [text_row0, text_row1, text_row2, text_row3, text_row4]

    true_label_5samples = [0, 1, 2, 3, 4]
    test_pred_class = []
    test_pred_3best_classes = []
    y_test_class = []

    for i in range(5):
        cropped_image_name = suppress_white_areas(DEST, list_images_5samples[i])
        cropped_image_name = DEST + cropped_image_name
        img_tensor = transform_image(cropped_image_name, target_size)

        str_tokens = preprocessing_text(list_text_5samples[i])
        txt_tensor = transform_text(str_tokens, text_size, tokenizer)
        three_best_classes = []
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
        test_pred_3best_classes.append(three_best_classes)

        y_test_class.append(true_label_5samples[i])

    for i in range(5):
        st.write(
            "Rakuten class : {0}  and  predicted class : {1}".format(
                name_class[y_test_class[i]], name_class[test_pred_class[i]]
            )
        )
        st.write(
            "next potential classes according to softmax decreasing probabilities : {0} and {1}".format(
                name_class[test_pred_3best_classes[i][1]],
                name_class[test_pred_3best_classes[i][2]],
            )
        )
        st.write("for associated text : ", list_text_5samples[i])
        st.text("and its related image : ")
        full_name = DEST + list_images_5samples[i]
        img_color = cv2.imread(full_name, cv2.IMREAD_COLOR)  # read the image
        fig_img = plt.figure(figsize=(20, 10))
        plt.imshow(img_color)
        st.pyplot(fig_img)
        st.text("")


def main():
    """
    function that is the main part of the program
    """
    st.title(
        "challenge Rakuten text and image multimodal DL classification Application"
    )
    st.markdown(
        "**In this notebook , a multimodal model based on tensorflow/keras framework is presented.**"
    )
    st.markdown(
        "**It has been developped in the scope of a Rakuten challenge whose objective is to classify among 27 classes products sold by this e-commerce major player each item described by texts and one image.**"
    )

    st.markdown("**Here are the main summarized steps :**")

    st.markdown(
        "*   **texts and images are preprocessed to become numerical data ready for model.**"
    )

    st.markdown(
        "*   **Then, a model using both data source informations has been designed. It is composed of two chains the first one encoding images based on pretrained VGG16 and the second one for the texts built on complementary embedding plus RNN GRU layers .**"
    )

    st.markdown(
        "*   **Then, extracted features from both data sources are concatenated to enter the multi Dense classifier with its output layer using softmax activation to predict for each product its probabilities between the 27 classes.**"
    )

    st.markdown(
        "*   **Finally, the model is evaluated using the test set to check its ability to generalize correct class predictions on data of new products.**"
    )
    st.text("")
    st.write("Python version :", sys.version)
    st.write("TensorFlow version :", tf.version.VERSION)
    st.write("numpy version :", np.__version__)
    st.text("")
    GPU, CPU, BESTDEV = set_best_device()
    st.markdown("**Read dataframe**")
    st.markdown("**Note : In the CSV, the text has already been processed**")
    st.markdown(
        "**translation in french language if necessary, stopwords, cleaning of html tags, tokenization**"
    )
    st.markdown(
        "**and for image a preprocessing has also been applied to suppress surrounding white zones**"
    )
    data_train = load_dataset()
    display_relationship_encoded_classes()
    st.markdown(
        "**After image preprocessing with openCV library to suppress surrounding white areas , one side effect has to be managed.**"
    )
    st.markdown(
        "**row of products associated with an image with a null size or a too small size (threshold set at 2 Kb) will be deleted**"
    )
    st.markdown("**as they can't be considered as meaningfull input for VGG16 model.**")
    # delete rows where images are smaller than 2k equal to noise rather than information
    data_train = delete_null_or_small_images(data_train)
    # cell specific for text preprocessing subpart
    # column designation_token_final is the result of preprocessing text cleaning (stopwords and so on) under list of tokens form
    # so, we transform it again into text string in designation_str_tokens that is now the "text features data" column
    # to prepare the association between each word and a integer number (index in dictionary of full vocabulary describing products)
    data_train = create_col_designation_str_tokens(data_train)
    text_length_explanation_for_model(data_train)
    st.markdown(
        "**Respect of separation of data to keep completely unknown part of them to trained model**"
    )
    st.markdown(
        "**ratio 70pct training - 20pct for validation and 10pct for test/evaluation**"
    )
    df_train, df_valid, df_test = prepare_train_valid_test_df(data_train)
    st.markdown("**Tensorflow/keras tokenizer created and trained on corpus**")
    st.markdown(
        "**to be used to convert into numerical numbers each product sentences(texts) put in batchs**"
    )
    tokenizer = train_tokenizer(df_train)
    st.markdown(
        "#Use case : Image + text handling with classifier composed of 2 Dense layers and a BatchNormalization between"
    )
    st.markdown("* **Image chain : pretrained VGG16 model**")
    st.markdown("* **Text chain : embedding + GRU layers**")
    st.markdown(
        "**Note : No customized loss function and metrics used to reduce possible side effect of imbalanced classes**"
    )
    st.markdown(
        "**as small classes have normal f1-score in the mean so others stronger explanations to be searched for few classes with the weakest f1-score**"
    )
    multimodal_model = instanciate_model()
    train_model(multimodal_model, tokenizer, df_train, df_valid)
    evaluation_model(df_test)
    predict_model(tokenizer)
    st.markdown("**In conclusion :**")
    st.markdown(
        "**prediction errors done by the model are quite easy to understand for those who have study Rakuten classification**"
    )
    st.markdown("**as some classes are very close or even overlapped.**")
    st.markdown(
        "**pratically no added value to use image encoded features as ML or DL based on encoded text ONLY are around 81pct of classification performance**"
    )
    st.markdown("**but for study purpose, it was fantastic**")
    st.text("")
    st.header("end of Rakuten classification model development script")


main()
