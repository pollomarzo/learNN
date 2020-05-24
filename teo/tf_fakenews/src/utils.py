import os
import string
import re
import nltk
from nltk.corpus import stopwords
import csv
import pandas as pd
from tqdm import trange

# consider subs in
# https://medium.com/@sabber/classifying-yelp-review-comments-using-lstm-and-word-embeddings-part-1-eb2275e4066b
# or https://github.com/pmsosa/CS291K/blob/master/batchgen.py

MAX_NB_WORDS = 5000
CURRENT_DIR = os.path.dirname(__file__)
stemmer = nltk.stem.SnowballStemmer('english')


def clean_str(src):
    """
    Cleaning of dataset, small version
    """
    src = re.sub(r"\\", "", string)
    src = re.sub(r"\'", "", string)
    src = re.sub(r"\"", "", string)
    return src.strip().lower()


def clean_text(text):
    """
    Cleans a single sentence
    """

    # Remove puncuation
    text = text.translate(string.punctuation)

    # Convert words to lower case and split them
    text = text.lower().split()

    # Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)  # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"didn t", "did not", text)
    # Stemming
    text = text.split()
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text
    # use it with pandas series with text = text.map(lambda x: clean_text(x))


def clean_and_save(data, current_dir, clean_file_dir="../clean_data/clean_train.csv"):
    """
    cleans training data and saves it to clean_file_dir
    """
    print("I see you've never run this before! Cleaning up training data")
    print("This may take a while...")
    texts = data[0]
    clean = [clean_text(texts[i]) for i in trange(len(data[0]))]
    labels = data[1]
    total = [[clean[i], labels[i]] for i in range(len(clean))]
    with open(os.path.join(current_dir, clean_file_dir), "w", newline="") as f:
        print("\tWriting to file...")
        writer = csv.writer(f)
        writer.writerow(('data', 'labels'))
        writer.writerows(total)
        print("\tDone!")


def prepare_workspace(current_dir, DIRECTORIES):
    """
    creates directories as specified, or if they exist does nothing
    """
    os.makedirs(os.path.join(
        current_dir, DIRECTORIES.CLEAN_DATA_DIR), exist_ok=True)
    os.makedirs(os.path.join(
        current_dir, DIRECTORIES.MODEL_DIR), exist_ok=True)
    os.makedirs(os.path.join(
        current_dir, DIRECTORIES.TRAINING_HISTORY_DIR), exist_ok=True)
    os.makedirs(os.path.join(
        current_dir, DIRECTORIES.TEST_RESULT_DIR), exist_ok=True)
