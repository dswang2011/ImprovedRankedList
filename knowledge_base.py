# -*- coding: utf-8 -*-
from copy import copy
from utils import stem_words
def get_prepared_KB(file_prepared_KB, stemmed = True):
    phrase2candidates = {}
    with open(file_prepared_KB,'r',encoding = 'utf-8') as f:
        p = ''
        candi_list = []
        for line in f:
            if line.startswith('=='):
                # process last one
                if p != '':
                    phrase2candidates[p] = copy(candi_list)

                # process next one
                strs = line.split('\t')
                p = strs[1].strip()
                candi_list.clear()  # clear the list
            else:
                strs = line.split('\t')
                if len(strs)>1:
                    explain = strs[1].replace("[Wikipedia]","")
                    explain = explain.replace("[Wiktionary]","")
                    if stemmed:
                        explain = stem_words(explain)
                    # print(explain)
                    candi_list.append(explain)

        if p != '':
            phrase2candidates[p] = copy(candi_list)
    # print(phrase2candidates)
    return phrase2candidates


# test case:print(explain)
# knowledge_base = get_prepared_KB(file_prepared_KB)
# phrases,scenarios = read_test_data(test_data)
# print(knowledge_base['dog meat'])
# for phrase, scenario in zip(phrases, scenarios):
#     candi_list = get_candidate_pages(phrase,knowledge_base)
#     print(phrase,candi_list)
    # print()


# print(phrases)
# print(scenarios)
# for p in phrases:
# 	candi_list = get_candidate_pages(p,knowledge_base)
# 	print(p,":",candi_list)
