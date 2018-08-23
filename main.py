from functions import *
from indri import *
import nltk
# nltk.download('wordnet')
def rank_list_comp(p, corpus, knowledge_base, scenario, context_type = 'wordvec', threshold = 0.3):
    # p: orginal phrase
    # original_ranked_list: L_p
    # perturbed_ranked_lists: \hat(L)_p
    # term_context_dict: (term, context) pairs


    # Get phrase contexts in a query or in a certain context
    phrase_context = get_context(scenario, p)
    index_dir_path = "E:/qiuchi/index/index_clueweb12"
    index = IndriAPI(index_dir_path)

    if 'adapt_with_knowledge_base':
        # Get phrase diambiguation pages in the knowledge base
        candidate_context_list = get_candidate_pages(phrase, knowledge_base)

        # Find all (matched contexts, matching scores) above a given threshold
        matched_contexts_dic = get_matched_contexts(phrase_context, candidate_context_list, threshold)

        # Generate the contexts of the original phrase based on matched candidate pages
        phrase_context = compute_updated_context(matched_contexts_dic)
############################################################################

    perturbed_phrase_list = get_perturbed_phrases(p)

    # get context of perturbed phrases
    perturbed_phrase_context_list = []
    for perturbed_phrase in perturbed_phrase_list:
        context_list = index.get_context(corpus, perturbed_phrase)
        if context_type == 'tfidf':
            context_rep = get_context_TFIDF(context_list, index)
        elif context_type == 'word_embedding':
            context_rep = get_context_WordEmbedding(context_list, lookup_table)

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

def compute_updated_context(matched_contexts):
    output_context = []
    for context in matched_contexts:
        score = matched_contexts[context]

    return output_context

def get_candidate_pages(phrase, knowledge_base):
    candidate_pages = []
    return candidate_pages

def get_matched_contexts(phrase_context, candidate_contexts_list, threshold):
    matched_contexts_dic = {}
    for candidate_context in candidate_contexts_list:
        score = ranked_list_similarity(phrase_context, candidate_context)
        if score > threshold:
            matched_contexts_dic[candidate_context] = score

    return matched_contexts_dic


