def rank_list_comp(p, corpus, knowledge_base, p_scenario = None, threshold = 0.3):
    # p: orginal phrase
    # original_ranked_list: L_p
    # perturbed_ranked_lists: \hat(L)_p
    # term_context_dict: (term, context) pairs

    terms = p.split()

    term_context_dict = {}
    # The lines below are what we propose

    # Get phrase contexts in a query or in a certain context
    scenario = p_scenario
    if scenario == None:
        scenario = corpus
    phrase_context = get_context(scenario, p)

    if 'adapt_with_knowledge_base':
        # Get phrase diambiguation pages in the knowledge base
        candidate_context_list = get_candidate_pages(phrase, knowledge_base)

        # Find all (matched contexts, matching scores) above a given threshold
        matched_contexts_dic = get_matched_contexts(phrase_context, candidate_context_list, threshold)

        # Generate the contexts of the original phrase based on matched candidate pages
        phrase_context = compute_updated_context(matched_contexts_dic)
############################################################################

    # get context of perturbed phrases
    perturbed_phrase_context_list = []
    for term in terms:
        synonym_terms = get_synonym_terms(knowledge_base, term) #get synonym terms from knowledge base

        for synonym_term in synonym_terms:
            perturbed_phrase = p.replace('term',synonym_term)
            ranked_list = {}
            context = get_context(corpus, perturbed_phrase)
            perturbed_phrase_context_list.append(context)


    # compute compositionality scores
    score = 0
    for perturbed_phrase_list in perturbed_phrase_context_list:
        score = score + ranked_list_similarity(perturbed_phrase_list, phrase_context)

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
        score = ranked_list_similarity(phrase_context, candidate_context, 1)
        if score > threshold:
            matched_contexts_dic[candidate_context] = score

    return matched_contexts_dic


def get_context(corpus, phrase):
    term_weight_dict = {}
    return term_weight_dict




def ranked_list_similarity(ranked_list_1, ranked_list_2, weight):
    score = 0
    return score

def update_ranked_list(ranked_list, context):

    return


def get_synonym_terms(knowledge_base, term):
    term_list = []
    return(term_list)
