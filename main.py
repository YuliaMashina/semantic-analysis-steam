from model import *
from utils import *
import numpy as np
from preprocess import *
import pandas as pd
import pickle
import os, sys
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore', category=Warning)

import argparse


def find_review (words_of_interest_arr, sentences_arr, original_sentences, return_preprocessed_sentences = False):
    words_of_interest_arr = pd.Series(np.array(words_of_interest_arr)).fillna('')
    sentences_arr = pd.Series(np.array(sentences_arr)).fillna('')
    n = 3
    reviews_found = []
    sentence_word_counter = np.zeros(len(sentences_arr))
    idx = 0
    for sentence in sentences_arr:
        for word in words_of_interest_arr:
            if word in sentence:
                sentence_word_counter[idx] += 1
        idx += 1

    indices = (-sentence_word_counter).argsort()[:n]
    if return_preprocessed_sentences:
        reviews_found.append([])
        reviews_found.append([])
        for index in range(n):
            reviews_found[0].append(sentences_arr[int(indices[index])])
            reviews_found[1].append(original_sentences[int(indices[index])])
        return reviews_found
    for index in range(n):
        reviews_found.append(original_sentences[int(indices[index])])
    found_reviews_to_write = ''
    for review_index in range(len(reviews_found)):
        found_reviews_to_write += str(review_index + 1) + '. '
        found_reviews_to_write += reviews_found[review_index]
        found_reviews_to_write += '\n\n'
    return found_reviews_to_write


def get_rws():
    data = pd.read_csv(r'C:\Users\Yulia\Downloads\contextual_topic_identification\data\steam_reviews.csv')
    data = data.fillna('')
    rws = data.review
    return rws


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(os.getcwd())

    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', default=r'C:\Users\Yulia\Downloads\contextual_topic_identification\data\steam_reviews.csv')
    parser.add_argument('--ntopic', default=10)
    parser.add_argument('--method', default='LDA_BERT')  # CHANGE FOR ANOTHER METHOD !!!
    parser.add_argument('--samp_size', default=10000)
    args = parser.parse_args()

    rws = get_rws()
    sentences, token_lists, idx_in = preprocess(rws, samp_size=int(args.samp_size))
    sentences_for_check = []
    for i in range(len(rws)):
        sentence = preprocess_sentence(rws[i])
        sentences_for_check.append(sentence)

    WORDS_OF_INTEREST = ['bug', 'problem', 'server', 'cheat', 'hack']  # WORDS_OF_INTEREST

    found_reviews = find_review(pd.Series(np.array(WORDS_OF_INTEREST)), pd.Series(np.array(sentences_for_check)).fillna(''), rws, False)
    file3 = open(r"C:\Users\Yulia\Downloads\contextual_topic_identification\example_reviews-overall.txt", "w", encoding='utf-8')
    file3.write(str(found_reviews))
    file3.close()

    tm = Topic_Model(k = int(args.ntopic), method = str(args.method))
    with tf.device('/gpu:0'):
        tm.fit(sentences, token_lists)
    # Evaluate using metrics
    with open(r"C:\Users\Yulia\Downloads\contextual_topic_identification\docs\saved_models\{}.file".format(tm.id), "wb") as f:
        try:
            pickle.dump(tm, f, pickle.HIGHEST_PROTOCOL)
        except:
            print('cant pickle weakref objects')
    #file0 = open(r"C:\Users\Yulia\Downloads\contextual_topic_identification\corpus.txt", "w", encoding='utf-8')
    #file0.write(str(tm.corpus))
    #file0.close()
    #file1 = open(r"C:\Users\Yulia\Downloads\contextual_topic_identification\dictionary.txt", "w", encoding='utf-8')
    #file1.write(str(tm.dictionary))
    #file1.close()
    #file2 = open(r"C:\Users\Yulia\Downloads\contextual_topic_identification\sentences.txt", "w", encoding='utf-8')
    #file2.write(str(sentences))
    #file2.close()

    print('Coherence c_v:', get_coherence(tm, token_lists, 'c_v'))
    print('Silhouette Score:', get_silhouette(tm))
    print('Coherence u_mass:', get_coherence(tm, token_lists, 'u_mass'))
    # visualize and save img
    visualize(tm)
    if tm.method != 'LDA':
        for i in range(tm.k):
            get_wordcloud(tm, token_lists, i)
