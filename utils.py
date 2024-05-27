import pandas as pd
import numpy as np

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import string

import os

def read_data(input_filepath):
    df_ = pd.read_csv(input_filepath)
    print(df_.head())
    
    return df_

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def remove_duplicates(input_df):
    num_duplicates = input_df.duplicated().sum()
    df_cleaned_ = input_df.drop_duplicates()

    return num_duplicates, df_cleaned_

def remove_null(input_df):
    init_row_cnt = len(input_df)
    df_cleaned_ = input_df.dropna()
    final_row_cnt = len(df_cleaned_)

    rows_removed = init_row_cnt - final_row_cnt

    return df_cleaned_, rows_removed

def data_quali_check(input_df):
    df_shape_ = input_df.shape
    print("Data Shape: ", df_shape_)

    df_dtypes_ = input_df.dtypes
    print("Data Types: ", df_dtypes_)

    cleaned_df_after_remove_null, row_removed_cnt = remove_null(input_df)
    print("Number of rows being removed due to NA: ", row_removed_cnt)


    duplicates_cnt, cleaned_df_ = remove_duplicates(cleaned_df_after_remove_null)
    print("Number of rows being removed due to duplication: ", duplicates_cnt)

    cleaned_df_shape_ = cleaned_df_.shape
    print("Final Data Shape: ", cleaned_df_shape_)

    return cleaned_df_

def remove_punctuation(df_input, columns):
    translator = str.maketrans("", "", string.punctuation)
    for col in columns:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str).apply(lambda x: x.translate(translator))
        else:
            raise ValueError(f"Column '{col}' does not exist in the df")
    return df_input

def remove_urls(df_input, columns):
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    remove_urls = lambda x: url_pattern.sub("", x)
    for col in columns:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str).apply(remove_urls)
        else:
            raise ValueError(f"Column '{col}' does not exist in the df")
    return df_input

def remove_emojis(df_input, columns):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    remove_emojis = lambda x: emoji_pattern.sub(r"", x)
    for col in columns:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str).apply(remove_emojis)
        else:
            raise ValueError(f"Column '{col}' does not exist in the df")
    return df_input

def remove_symbols(df_input, columns):
    symbol_pattern = re.compile(r"[^\w\s]")
    remove_symbols = lambda x: symbol_pattern.sub("", x)
    for col in columns:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str).apply(remove_symbols)
        else:
            raise ValueError(f"Column '{col}' does not exist in the df")
    return df_input

def remove_non_ascii(df_input, columns):
    remove_non_ascii = lambda x: "".join(char for char in x if ord(char) < 128)
    for col in columns:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str).apply(remove_non_ascii)
        else:
            raise ValueError(f"Column '{col}' does not exist in the df")
    return df_input

def remove_numeric(df_input, columns):
    remove_numeric = lambda x: "".join(char for char in x if not char.isdigit())
    for col in columns:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str).apply(remove_numeric)
        else:
            raise ValueError(f"Column '{col}' does not exist in the df")
    return df_input

def lowercasing(df_input, columns):
    lowercase = lambda x: x.lower()
    for col in columns:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str).apply(lowercase)
        else:
            raise ValueError(f"Column '{col}' does not exist in the df")
    return df_input

def remove_stopwords(df_input, columns):
    stop_words = set(stopwords.words("english"))
    remove_stopwords = lambda tokens: [word for word in tokens if word not in stop_words]
    for col in columns:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str).apply(word_tokenize).apply(remove_stopwords).apply(" ".join)
        else:
            raise ValueError(f"Column '{col}' does not exist in the df")
    return df_input

def lemmatisation(df_input, columns):
    lemmatiser = WordNetLemmatizer()
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    lemmatise = lambda tokens: [lemmatiser.lemmatize(word, get_wordnet_pos(word)) for word in tokens]
    for col in columns:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str).apply(word_tokenize).apply(lemmatise).apply(" ".join)
        else:
            raise ValueError(f"Column '{col}' does not exist in the df")
    return df_input

def stemming(df_input, columns):
    stemmer = SnowballStemmer("english")
    stem = lambda tokens: [stemmer.stem(word) for word in tokens]
    for col in columns:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str).apply(word_tokenize).apply(stem).apply(" ".join)
        else:
            raise ValueError(f"Column '{col}' does not exist in the df")
    return df_input

def text_preprocessing_combi(df_input, columns):
    df_ = remove_punctuation(df_input, columns)
    df_ = remove_urls(df_, columns)
    df_ = remove_emojis(df_, columns)
    df_ = remove_symbols(df_, columns)
    df_ = remove_non_ascii(df_, columns)
    df_ = remove_numeric(df_, columns)
    df_ = lowercasing(df_, columns)
    df_ = remove_stopwords(df_, columns)
    df_ = lemmatisation(df_, columns)
    df_ = stemming(df_, columns)
    return df_

def export_to_csv(df_input, output_filepath):
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df_input.to_csv(output_filepath, index=False)
    print(f"Df successfully exported to {output_filepath}")