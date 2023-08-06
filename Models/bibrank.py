# coding:utf-8
import math
#import stemming.porter2 as porter
import numpy as np
import copy
from collections import Counter
from mediawiki import MediaWiki
from nltk.stem import WordNetLemmatizer
from nltk import PorterStemmer

def get_cat(text):
    try:

        wikipedia = MediaWiki()
        t= wikipedia.search(text)[0]
        p = wikipedia.page(t)
        out=[]
        for cat in  p.categories:
            out.append(cat.lower())

        return out
    except:
        return []

def get_weights(keywords_list):
    words = {}
    for keywords in keywords_list:
        for keyword in keywords:
            try:
                words[keyword] += 1
            except:
                words[keyword] = 1
    max_key = max(words, key=words.get)
    factor = words[max_key] / 1

    for k in words:
        words[k] = words[k] / factor
    return words


def bibrank(text, tokenizer, alpha=0.85, window_size=6, num_keyphrase=10, lang="en",weights={},wiki=False):
    """
    Position Features are based on the work done in PositionRank
    Original paper is here: http://aclweb.org/anthology/P/P17/P17-1102.pdf
    """
    lemmatizer = WordNetLemmatizer()
    if lang == "en":
        stem =  lemmatizer.lemmatize #porter.stem #
    else:
        stem = lambda word: word
    
    wiki_list=[]

    try:
        title,sentence=text[0]," ".join(text)
    except:
        title, sentence =""," ".join(text)

    
    if wiki:
       wiki_list =  get_cat(title)

    # origial words(=no stemming) and phrase list
    original_words, phrases = tokenizer.tokenize(sentence)
    # stemmed words
    stemmed_word = [stem(word) for word in original_words]
    unique_word_list = set([word for word in stemmed_word])
    n = len(unique_word_list)

    adjancency_matrix = np.zeros((n, n))
    word2idx = {w: i for i, w in enumerate(unique_word_list)}
    p_vec = np.zeros(n)
    # store co-occurence words
    co_occ_dict = {w: [] for w in unique_word_list}

    # 1. initialize  probability vector
    for i, w in enumerate(stemmed_word):
        # add position score
        p_vec[word2idx[w]] += float(1 / (i+1))
        for window_idx in range(1, math.ceil(window_size / 2)+1):
            if i - window_idx >= 0:
                co_list = co_occ_dict[w]
                co_list.append(stemmed_word[i - window_idx])
                co_occ_dict[w] = co_list

            if i + window_idx < len(stemmed_word):
                co_list = co_occ_dict[w]
                co_list.append(stemmed_word[i + window_idx])
                co_occ_dict[w] = co_list

    # 2. create adjancency matrix from co-occurence word
    for w, co_list in co_occ_dict.items():
        cnt = Counter(co_list)
        for co_word, freq in cnt.most_common():
            adjancency_matrix[word2idx[w]][word2idx[co_word]] = freq

    adjancency_matrix = adjancency_matrix / adjancency_matrix.sum(axis=0)
    
    p_vec = p_vec / p_vec.sum()
    # principal eigenvector s
    s_vec = np.ones(n) / n


    # threshold
    lambda_val = 1.0
    loop = 0
    # compute final principal eigenvector
    while lambda_val > 0.001:
        next_s_vec = copy.deepcopy(s_vec)
        for i, (p, s) in enumerate(zip(p_vec, s_vec)):
            next_s = (1 - alpha) * p + alpha * (weight_total(adjancency_matrix, i, s_vec))
            next_s_vec[i] = next_s
        lambda_val = np.linalg.norm(next_s_vec - s_vec)
        s_vec = next_s_vec
        loop += 1
        if loop > 100:
            break

    # score original words and phrases
    word_with_score_list = [(word, s_vec[word2idx[stem(word)]]) for word in original_words]

    for phrase in phrases:
        try:
            total_score = sum([s_vec[word2idx[stem(word)]] for word in phrase.split("_")])
        except:
            pass

        for word in phrase.split("_"):
            try:
                total_score+=weights[word.lower()]
            except:
                pass
            
            for cat  in wiki_list:
                if word.lower() in cat:
                    total_score*=2
        try:
            word_with_score_list.append((phrase, total_score))
        except:
            word_with_score_list.append((phrase, 0.0))


    
    sort_list = np.argsort([t[1] for t in word_with_score_list])
    

    keyphrase_list = []

    
    # if not check stemmed keyphrase, there are similar phrases in keyphrase list
    # i.e. "neural network" and "neural networks" in list
    stemmed_keyphrase_list = []
    for idx in reversed(sort_list):
        keyphrase = word_with_score_list[idx][0]
        stemmed_keyphrase = " ".join([stem(word) for word in keyphrase.split("_")])
        if not stemmed_keyphrase in stemmed_keyphrase_list:
            keyphrase=keyphrase.replace("_", " ")
            keyphrase_list.append((keyphrase,word_with_score_list[idx][1]))
            stemmed_keyphrase_list.append(stemmed_keyphrase)
        if len(keyphrase_list) >= num_keyphrase:
            break

    return keyphrase_list


def weight_total(matrix, idx, s_vec):
    """Sum weights of adjacent nodes.

    Choose 'j'th nodes which is adjacent to 'i'th node.
    Sum weight in 'j'th column, then devide wij(weight of index i,j).
    This calculation is applied to all adjacent node, and finally return sum of them.

    """
    return sum([(wij / matrix.sum(axis=0)[j]) * s_vec[j] for j, wij in enumerate(matrix[idx]) if not wij == 0])
