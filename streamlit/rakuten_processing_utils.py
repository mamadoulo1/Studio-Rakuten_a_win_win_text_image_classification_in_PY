"""
module listing processing functions of Rakuten projet
"""
# library for Regular Expression
import re

from pathlib import Path

# Web UI for demo
import streamlit as st

# Data Science useful python librairies
import numpy as np


# DL framework
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Embedding, GRU
from tensorflow.keras.layers import BatchNormalization, Activation, Concatenate
from tensorflow.keras import activations
from tensorflow.keras.applications.vgg16 import VGG16

from rakuten_constants import *

# class and functions used in the notebook linked with model


def create_tokenizer(text):
    """function that create a tokenizer and applies it on corpus (texts of Rakuten products of training set ONLY )
    input  : Serie of training dataframe containing text in string format
    return : tokenizer object
    """
    # tokenizer definition
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features)
    # update tokenizer dictionary
    tokenizer.fit_on_texts(text)

    # Store dictionary associating word and index in variable word2idx
    word2idx = tokenizer.word_index

    # Store dictionary associating index and word in variable idx2word
    idx2word = tokenizer.index_word

    # Store dictionary size
    vocab_size = tokenizer.num_words

    return tokenizer


# creation of customized Data Generator used for building batchs
class CustomDataGen(tf.keras.utils.Sequence):
    """
    Class used to generate batch of images and texts for the model
    """

    def __init__(
        self,
        df,
        tokenizer,
        x_col_img,
        x_col_txt,
        y_col,
        batch_size,
        input_size=target_size,
        input_text_size=text_size,
        shuffle=False,
    ):

        self.df = df.copy()
        self.tokenizer = tokenizer
        self.x_col_img = x_col_img
        self.x_col_txt = x_col_txt
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.input_text_size = input_text_size
        self.shuffle = shuffle

        self.len_df = len(self.df)
        self.n_classes = df[y_col["class_onehot"]].nunique()

    def on_epoch_end(self):
        """
        nothing scheduled for end of epoch
        """
        pass

    def __get_input_img(self, path, target_size):

        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)

        image_arr = tf.image.resize(image_arr, (target_size[0], target_size[1])).numpy()
        # normalization of values between 0 and 1 for Deep Learning gradient descent backpropagation
        return image_arr / 255.0

    def __get_input_txt(self, text, size, tokenizer):

        x_text_int = tokenizer.texts_to_sequences([text])
        # pad or truncate list of integers
        x_text = tf.keras.preprocessing.sequence.pad_sequences(
            x_text_int, maxlen=size, padding="post", truncating="post"
        )
        # only return the list not a list of one list !!!!!!
        return x_text[0]

    def __get_output(self, label, num_classes):
        return tf.keras.utils.to_categorical(label, num_classes=num_classes)

    def __get_data(self, batches):
        # Generates data containing batch_size samples

        image_batch = batches[self.x_col_img["nom_img"]]

        text_batch = batches[self.x_col_txt["designation_str_tokens"]]

        class_batch = batches[self.y_col["class_onehot"]]

        x_batch_img = np.stack(
            [self.__get_input_img(x, self.input_size) for x in image_batch]
        )

        x_batch_txt = np.stack(
            [
                self.__get_input_txt(x, self.input_text_size, self.tokenizer)
                for x in text_batch
            ]
        )

        y_batch = np.stack([self.__get_output(y, self.n_classes) for y in class_batch])

        return x_batch_img, x_batch_txt, y_batch

    def __getitem__(self, index):

        # index is managed by fit method as it is a subclass of Sequence Class
        batches = self.df[index * self.batch_size : (index + 1) * self.batch_size]
        x_img, x_txt, y_class = self.__get_data(batches)
        return [x_img, x_txt], y_class

    def __len__(self):
        return self.len_df // self.batch_size


def transform_image(path, target_size):
    """function that converts an image and put it into a tensor 4D"""
    image = tf.keras.preprocessing.image.load_img(path)
    image_arr = tf.keras.preprocessing.image.img_to_array(image)

    image_arr = tf.image.resize(image_arr, (target_size[0], target_size[1])).numpy()
    image_arr = np.expand_dims(
        image_arr, axis=0
    )  # img 3D becoming a tensor 4D (one dimension added)

    return image_arr / 255.0


def transform_text(text_in, size, tokenizer):
    """function that converts an text and put it into a tensor 2D using tokenizer"""
    x_text_int = tokenizer.texts_to_sequences([text_in])
    # pad or truncate list of integers
    text = tf.keras.preprocessing.sequence.pad_sequences(
        x_text_int, maxlen=size, padding="post", truncating="post"
    )
    # only return the list not a list of one list !!!!!!
    return text


def create_col_designation_str_tokens(dataf):
    """function that adds a string column named designation_str_tokens using column designation_token_final of received dataframe
    input  : dataframe with column designation_token_final being a list of tokens
    return : dataframe with new column designation_str_tokens
    """
    df = dataf
    df["designation_str_tokens"] = ""
    for i in range(len(df)):
        df["designation_str_tokens"].iloc[i] += "".join(
            df["designation_token_final"].iloc[i]
        )

    df["designation_str_tokens"] = df["designation_str_tokens"].map(
        lambda x: re.sub(r",", "", x)
    )
    df["designation_str_tokens"] = df["designation_str_tokens"].map(
        lambda x: re.sub(r"'", "", x)
    )
    df["designation_str_tokens"] = df["designation_str_tokens"].map(
        lambda x: re.sub(r"\[", "", x)
    )
    df["designation_str_tokens"] = df["designation_str_tokens"].map(
        lambda x: re.sub(r"\]", "", x)
    )

    return df


def create_col_class_onehot(dataf):
    """function that adds a column named class_onehot using column prdtypecode of received dataframe
    input  : dataframe with column prdtypecode being the Rakuten classes
    return : dataframe with new column class_onehot
    """
    df = dataf
    df["class_onehot"] = 0
    for i in range(len(df)):
        index = list_class.index(df["prdtypecode"].iloc[i])
        df["class_onehot"].iloc[i] = list_class_onehot[index]

    return df


def delete_null_or_small_images(dataf):
    """function that deletes rows in received dataframe associated with an image with null size or smaller than 2Kb
    input  : dataframe with column nom_img
    return : dataframe with samples respecting image size criteria : bigger than 2 Kb
    """
    df = dataf
    list_products_to_be_deleted_due_to_too_small_image = []
    nbr_to_be_deleted = 0
    for i in range(len(df)):
        fname = df["nom_img"].iloc[i]
        file_size = Path(fname).stat().st_size
        if file_size < 2048:
            # print("The file {0} has a too small size of {1} bytes ".format(fname,file_size))
            list_products_to_be_deleted_due_to_too_small_image.append(i)
            nbr_to_be_deleted += 1

    # rows deletion
    st.text("")
    st.write(
        "number of deleted rows due to image size smaller than 2K :", nbr_to_be_deleted
    )
    st.write("shape before cleaning :", df.shape)
    df = df.drop(list_products_to_be_deleted_due_to_too_small_image)
    st.write(
        "new shape of dataset after suppressed samples due to image preprocessing  : ",
        df.shape,
    )

    return df


# multimodal DL model definition
def multimodal_vgg16_emb_gru(target_size, text_size, max_features):
    """
    This function creates a multimodal model based on pretrained VGG16 model for extracting image features
    and on keras embedding and GRU layers for extracting text features
    then concatenated to be passed to the Dense classifier generating the probabilities of the 27 classes
    """

    # VGG16 model used without its internal classifier
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=target_size)
    # Freezer les couches du VGG16
    for layer in base_model.layers:
        layer.trainable = False

    image_input = tf.keras.Input(shape=target_size, name="image_input")
    features_img = base_model(image_input)
    encoded_image = Flatten()(features_img)

    text_input = tf.keras.Input(shape=(text_size), name="text_input")
    embedded_text = Embedding(max_features + 1, 640)(text_input)
    features_txt = GRU(640)(embedded_text)
    encoded_txt = Flatten()(features_txt)

    # Concatenate both encoded images and text and pass through the classification layer.
    # Normalization is mandatory for Neural Network with numerous hidden layers
    # but this design pattern is anyway useful to ease weights calculation during descent gradient backpropagation

    concatenated = Concatenate()([encoded_image, encoded_txt])
    classifier_lay1 = Dense(64)(concatenated)
    classifier_lay2 = BatchNormalization()(classifier_lay1)
    classifier_lay3 = Activation(activations.relu)(classifier_lay2)
    classifier_lay4 = Dropout(0.2)(classifier_lay3)
    outputs = Dense(27, activation="softmax")(classifier_lay4)

    multimodal_model = Model([image_input, text_input], outputs)
    # summarize layers of the model for notebook (as print function is default)
    # multimodal_model.summary())
    stringlist = []
    multimodal_model.summary(print_fn=lambda x: stringlist.append(re.sub(r"#", "", x)))
    short_model_summary = "\n".join(stringlist)
    st.text("model summary :")
    st.write(short_model_summary)

    return multimodal_model
