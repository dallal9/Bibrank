'''
# Load your usual SpaCy model (one of SpaCy English models)
import spacy
nlp = spacy.load('en_core_web_sm')

# Add neural coref to SpaCy's pipe
import neuralcoref
neuralcoref.add_to_pipe(nlp)

print("test")
# You're done. You can now use NeuralCoref as you usually manipulate a SpaCy document annotations.
doc = nlp(u'My sister has a dog. She loves him./n this is it')

print(doc._.has_coref)
doc._.has_coref
print(doc._.coref_resolved)
'''
import pke

# define the valid Part-of-Speeches to occur in the graph
pos = {'NOUN', 'PROPN', 'ADJ'}

# define the grammar for selecting the keyphrase candidates
grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"

# 1. create a PositionRank extractor.
extractor = pke.unsupervised.PositionRank()

# 2. load the content of the document.
extractor.load_document(input='C:/Users/dallal/Documents/GitHub/ake-datasets/datasets/KDD/src/kpdata/KDD/abstracts/0',
                        language='en',
                        normalization=None)

# 3. select the noun phrases up to 3 words as keyphrase candidates.
extractor.candidate_selection(grammar=grammar,
                              maximum_word_number=3)

# 4. weight the candidates using the sum of their word's scores that are
#    computed using random walk biaised with the position of the words
#    in the document. In the graph, nodes are words (nouns and
#    adjectives only) that are connected if they occur in a window of
#    10 words.
extractor.candidate_weighting(window=10,
                              pos=pos)

# 5. get the 10-highest scored candidates as keyphrases
keyphrases = extractor.get_n_best(n=8)

print(keyphrases)

import pke
import json
from tqdm import tqdm
import codecs
from position_rank import position_rank
from tokenizer import StanfordCoreNlpTokenizer


import spacy
nlp = spacy.load('en_core_web_sm')

# Add neural coref to SpaCy's pipe
import neuralcoref
neuralcoref.add_to_pipe(nlp)


labels = []
conv = {'-LRB-':'(',
        '-RRB-':')',
        '-LCB-':'{',
        '-RCB-':'}',
        '-LSB-':'[',
        '-RSB-':']'}



def metric(g_truth,predicated):

    cnt_p= len(predicated)
    cnt_g= len(g_truth)
    correct=0.0
    for each in predicated:
        if each.lower() in g_truth:
            correct+=1
    try:      
        p=correct/cnt_p
    except:
        p=0.0

    try:
        r=correct/cnt_g
    except:
        r=0.0
    try:
        f1= (2*p*r)/(p+r)
    except:
        f1=0.0
    return p,r,f1
     

def extract(f):
    with codecs.open(f, 'r', 'utf-8') as f:

        lines = f.readlines()
        text = []
        for line in lines:
            tokens = line.strip().split()
            if len(tokens):
                untagged_tokens = [token.split('_')[0] for token in tokens]
                untagged_tokens = [conv[u] if u in conv else u for u in untagged_tokens]
                text.append(' '.join(untagged_tokens))
                #t = nlp(' '.join(untagged_tokens))
                #text.append(str(t._.coref_resolved))

    return text
with open('C:/Users/dallal/Documents/GitHub/ake-datasets/datasets/KDD/references/test.author.json', 'r',encoding='utf-8') as f:
    keyphrases_gold = json.load(f)

labels = list(keyphrases_gold.keys())

P,R,F1=0.0,0.0,0.0






for indx in tqdm(range(len(labels))):

    extractor = pke.unsupervised.PositionRank()
    label = labels[indx]
    f='C:/Users/dallal/Documents/GitHub/ake-datasets/datasets/KDD/test/'+str(label)+'.xml'
    f= 'C:/Users/dallal/Documents/GitHub/ake-datasets/datasets/KDD/src/kpdata/KDD/abstracts/'+str(label)
    text = extract(f)
    text[0]+=" "
    
    text = "".join(text)
    tokenizer = StanfordCoreNlpTokenizer("http://localhost", port = 9000)
    #extractor.load_document(input=f, language='en')

    #extractor.candidate_selection()

    #extractor.candidate_weighting(window=10)

    for n in [8]:#[2, 4, 6, 8]:  
        #keyphrases = extractor.get_n_best(n=n)

        #keyphrases_p=[]
        #for keyphrase in keyphrases:
        #    keyphrases_p.append(keyphrase[0])
        #keyphrases_p=position_rank(text, tokenizer,alpha=0.6, window_size=4,num_keyphrase=n)
        t= open("temp.txt","w",encoding="utf-8")
        t.write(text)
        t.close()
        # 2. load the content of the document.
        extractor.load_document(input='temp.txt',
                                language='en',
                                normalization=None)

        # 3. select the noun phrases up to 3 words as keyphrase candidates.
        extractor.candidate_selection(grammar=grammar,
                                    maximum_word_number=3)

        # 4. weight the candidates using the sum of their word's scores that are
        #    computed using random walk biaised with the position of the words
        #    in the document. In the graph, nodes are words (nouns and
        #    adjectives only) that are connected if they occur in a window of
        #    10 words.
        extractor.candidate_weighting(window=10,
                                    pos=pos)

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=8)

        keyphrases_p=[]
        for keyphrase in keyphrases:
            keyphrases_p.append(keyphrase[0])
        keyphrases_gold_p=[]
        for l in keyphrases_gold[label]:
            keyphrases_gold_p+=l


        p,r,f1=metric(keyphrases_gold_p,keyphrases_p)
        #print(keyphrases_gold_p,keyphrases_p)
        #input("cooooool .. ")
        P+=p
        R+=r
        F1+=f1
        if indx%50==0:
            print(R/(indx+1))
            print(P/(indx+1))
            print(F1/(indx+1))

       
print(P/len(labels))
print(R/len(labels))
print(F1/len(labels))