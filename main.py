from functions import *
from indri import *
import nltk
from readKB import *
from copy import copy
# nltk.download('wordnet')
def rank_list_comp(p, scenario,corpus_index, knowledge_base, matrix, word_list, context_type = 'tfidf', adapt_with_knowledge_base = True, threshold = 0.3):
    # p: orginal phrase
    # original_ranked_list: L_p
    # perturbed_ranked_lists: \hat(L)_p
    # term_context_dict: (term, context) pairs


    # Get phrase contexts in a query or in a certain context


    phrase_context_terms = get_phrase_context(scenario, p)
    if context_type == 'tfidf':
        phrase_context = get_context_TFIDF(phrase_context_terms, corpus_index)
    elif context_type == 'word_embedding':
        phrase_context = get_context_WordEmbedding(phrase_context_terms, matrix, word_list)

    if adapt_with_knowledge_base:
        # Get phrase diambiguation pages in the knowledge base
        candidate_context_list = get_candidate_pages(p, knowledge_base)

        # Find all (matched contexts, matching scores) above a given threshold
        matched_contexts_dic = get_matched_contexts(phrase_context, candidate_context_list, corpus_index, threshold = 0)

        # Generate the contexts of the original phrase based on matched candidate pages
        phrase_context = compute_updated_context(phrase_context,matched_contexts_dic, corpus_index)
############################################################################

    perturbed_phrase_list = get_perturbed_phrases(p)

    # get context of perturbed phrases
    perturbed_phrase_context_list = []
    for perturbed_phrase in perturbed_phrase_list:
        context_list = corpus_index.get_context_list(perturbed_phrase,window_size = 50)
        if context_type == 'tfidf':
            context_rep = get_context_TFIDF(context_list, corpus_index)
        elif context_type == 'word_embedding':
            context_rep = get_context_WordEmbedding(context_list, lookup_table,word_list)

        perturbed_phrase_context_list.append(context_rep)


    # compute compositionality scores
    score = 0
    if context_type == 'tfidf':
        for perturbed_phrase_list in perturbed_phrase_context_list:
            score = score + tfidf_similarity(perturbed_phrase_list, phrase_context)
    elif context_type == 'word_embedding':
        for perturbed_phrase_list in perturbed_phrase_context_list:
            score = score + context_similarity(perturbed_phrase_list, phrase_context)
    score = score/len(perturbed_phrase_context_list)
    return score

def compute_updated_context(phrase_context,matched_contexts,index):
    weight = 0.1
    output_context = phrase_context.copy()
    for term in output_context:
        output_context[term] = output_context[term]* weight
    total_score = 0
    for context in matched_contexts:
        total_score = total_score + matched_contexts[context]

    for context in matched_contexts:
        context_rep = get_context_TFIDF(context, index)
        for term in context_rep:
            if term in output_context:
                output_context[term] = output_context[term] + context_rep[term]*matched_contexts[context]/total_score* (1-weight)
            else:
                output_context[term] = context_rep[term]*matched_contexts[context]/total_score* (1-weight)

    return output_context


def get_matched_contexts(phrase_context, candidate_contexts_list, index,threshold):
    matched_contexts_dic = {}
    for candidate_context in candidate_contexts_list:
        if context_type == 'tfidf':
            candidate_context_rep = get_context_TFIDF(context_list, corpus_index)
        elif context_type == 'word_embedding':
            candidate_context_rep = get_context_WordEmbedding(context_list, lookup_table)
        score = tfidf_similarity(phrase_context, candidate_context_rep)
        if score > threshold:
            matched_contexts_dic[candidate_context] = score

    return matched_contexts_dic

if __name__ == '__main__':
    index_dir_path = "E:/qiuchi/index/index_clueweb12"
    index = IndriAPI(index_dir_path)
    file_prepared_KB = 'sampling_KB.txt'
    test_data = 'small_test.txt'
    path_to_vec = 'glove/glove.6B.50d.txt'
    matrix, word_list = form_matrix(path_to_vec)
    knowledge_base = get_prepared_KB(file_prepared_KB)
    phrases,scenarios,labels = read_test_data(test_data)
    for phrase, scenario in zip(phrases, scenarios):
        score= rank_list_comp(phrase,scenario, index,knowledge_base,matrix,word_list)
        print(phrase,score)
