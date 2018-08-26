from nltk.corpus import wordnet as wn
from indri import IndriAPI
from word_embedding import *
import numpy as np
import math
################## Qiuchi: ###################


# 1. for term weight, we comput TF_IDF with the corpus staistics (like TREC disks 4-5).


### 2. get context for a phrase from corpus ###
# input: a phrase or a single term
# output: 5-window cumulative context from corpus. weight should be added up for each word.

def get_context_TFIDF(context_list, index):
    collection_doc_count = index.get_collection_doc_count()
    term_weight_dict = {}
    for context in context_list:
        term_list = context.split()
        for term in term_list:
            idf = math.log10(collection_doc_count/(index.get_doc_frequency(term)+1))
            if term in term_weight_dict:
                term_weight_dict[term] = term_weight_dict[term]+idf
            else:
                 term_weight_dict[term] = idf
    for term in term_weight_dict:
        term_weight_dict[term] = term_weight_dict[term]/len(context_list)
    return term_weight_dict

def get_context_WordEmbedding(context_list, matrix, word_list):
    context_vector = np.zeros(shape = (1,matrix.shape[1]))
    for context in context_list:
        term_list = context.split()
        # term_vector
        index_list = []
        for term in term_list:
            if term in word_list:
                index_list.append(word_list.index(term))
        context_mean = np.mean(matrix[index_list,:],axis = 0)
        context_vector = context_vector + context_mean
        # print(context_vector.shape)
    context_vector = context_vector/len(context_list)	# 300 dimention
    return context_vector

def get_context_TF(phrase,scenario):
    term_weight_dict = {}
    for term in scenario:
        if term in term_weight_dict:
            term_weight_dict[term] = term_weight_dict[term]+1
        else:
            term_weight_dict[term] = 1

def get_phrase_context(scenario,p):
    return scenario

'''
3. get perturbated phrase for phrase p. The question is we should investigate how to filter out meanless ones (or top-k, k=5/6?).
input: phrase p
output： a list of phrases perturbed_phrase_list
'''
def get_perturbed_phrases(p):
    perturbed_phrase_list = []
    perturbed_phrases = []
    terms = p.split()
    for term in terms:
        synonym_terms = get_synonym_terms(term) #get synonym terms from knowledge base

        for synonym_term in synonym_terms:
            perturbed_phrase = p.replace(term,synonym_term)
            perturbed_phrase_list.append(perturbed_phrase)
    return perturbed_phrase_list

##

'''
4. get synomym, which is used by 3. Invoke WordNet api.
input: term
output: a list of all its synonyms
'''
def get_synonym_terms(term):
    synonym_term_list = []
    for ss in wn.synsets(term):
        synonym_term_list = synonym_term_list + ss.lemma_names()

    synonym_term_list = list(set(synonym_term_list))
    synonym_term_list.remove(term)
    return synonym_term_list


def tfidf_similarity(term_weight_dict1,term_weight_dict2):
    sum_square_weight_1 = 0
    sum_square_weight_2 = 0
    inner_product = 0
    for term in term_weight_dict1:
        sum_square_weight_1 =  sum_square_weight_1+ term_weight_dict1[term]**2
        if term in term_weight_dict2:
            inner_product = inner_product+ term_weight_dict2[term]* term_weight_dict1[term]

    for term in term_weight_dict2:
        sum_square_weight_2 =  sum_square_weight_2+ term_weight_dict2[term]**2

    return inner_product/np.sqrt(sum_square_weight_1*sum_square_weight_2)

def wordvec_similarity(wordvec1, wordvec2):
    return np.inner(wordvec1, wordvec2)/(np.linalg.norm(wordvec1)*np.linalg.norm(wordvec2))
if __name__ == '__main__':
    path_to_vec = 'glove/glove.6B.50d.txt'
    indri = IndriAPI('E:/qiuchi/index/index_clueweb12')
    context_list = ['It is a good day', 'today is a good day', 'black horse']
    matrix, word_list = form_matrix(path_to_vec)
    ranked_list = get_context_TFIDF(context_list, indri)
    # wordvec= get_context_WordEmbedding(context_list, matrix, word_list)

    # print(wordvec_similarity(wordvec,wordvec))
    print(tfidf_similarity(ranked_list,ranked_list))
    # print(indri.get_doc_frequency('ivory'))
    # print(get_perturbed_phrases(p))


