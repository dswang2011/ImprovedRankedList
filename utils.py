from nltk.corpus import wordnet as wn
from corpus_index import IndriAPI
from word_embedding import *
import numpy as np
import math
from scipy.stats import pearsonr
from nltk.stem.porter import *
import pandas as pd
import string
################## Qiuchi: ###################


# 1. for term weight, we comput TF_IDF with the corpus staistics (like TREC disks 4-5).

def get_phrase_context(scenario,p):
    return scenario

'''
3. get perturbated phrase for phrase p. The question is we should investigate how to filter out meanless ones (or top-k, k=5/6?).
input: phrase p
output： a list of phrases perturbed_phrase_list
'''
def get_perturbed_phrases(p,stem_words='False'):
    perturbed_phrase_list = []
    perturbed_phrases = []
    terms = p.split()
    for term in terms:
        synonym_terms = get_synonym_terms(term) #get synonym terms from knowledge base

        for synonym_term in synonym_terms:
            synonym_term = strip_punctuation(synonym_term)
            if stem_words in ['True','true','TRUE']:
                synonym_term = stem_words(synonym_term)
            perturbed_phrase = p.replace(term,synonym_term)
            perturbed_phrase_list.append(perturbed_phrase)
    if p in perturbed_phrase_list:
        perturbed_phrase_list.remove(p)
    return perturbed_phrase_list

'''
get synomym, which is used by 3. Invoke WordNet api.
input: term
output: a list of all its synonyms
'''
def get_synonym_terms(term):
    synonym_term_list = []
    for ss in wn.synsets(term):
        synonym_term_list = synonym_term_list + ss.lemma_names()

    synonym_term_list = list(set(synonym_term_list))
    if term in synonym_term_list:
        synonym_term_list.remove(term)
    return synonym_term_list


def read_test_data(test_data):
    phrases = []
    scenarios = []
    labels = []


    if test_data.endswith('csv'):
    # file_name = 'preprocessed_data.csv'
        df = pd.read_csv(test_data, names = ['phrase','scenario','cd_score'])
        phrases = df['phrase'].values
        scenarios = df['scenario'].values
        labels = df['cd_score'].values
    else:
        with open(test_data,'r',encoding = 'utf-8') as f:
            for line in f:
                strs = line.split('\t')
                if len(strs)>1:
                    phrases.append(strs[0].strip())
                    scenarios.append(strs[1].strip())
                    labels.append(strs[1].strip())
    return phrases,scenarios,labels
def strip_punctuation(text):
    return ''.join(c for c in text if c not in string.punctuation)

def pearson_correlation(x,y):
    return(pearsonr(x,y)[0])

def stem_words(text):
    stemmer = PorterStemmer()
    words = text.split()
    new_text = ''
    for word in words:
        new_text = new_text+ stemmer.stem(word) +' '

    new_text = new_text.strip()
    return new_text

if __name__ == '__main__':
    input_file = 'input/test_input.txt'
    phrases, scenarios, labels = read_test_data(input_file)
    for phrase,scenario,label in zip(phrases,scenarios,labels):
        print(phrase+'***'+scenario+'***'+label)
    # path_to_vec = 'glove/glove.6B.50d.txt'
    # indri = IndriAPI('E:/qiuchi/index/index_clueweb12')
    # context_list = ['It is a good day', 'today is a good day', 'black horse']
    # matrix, word_list = form_matrix(path_to_vec)
    # ranked_list = get_context_TFIDF(context_list, indri)
    # wordvec= get_context_WordEmbedding(context_list, matrix, word_list)

    # print(wordvec_similarity(wordvec,wordvec))
    # print(tfidf_similarity(ranked_list,ranked_list))
    # print(indri.get_doc_frequency('ivory'))
    # print(get_perturbed_phrases(p))


