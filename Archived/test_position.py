import pke
import json
from tqdm import tqdm
import codecs
from position_rank import position_rank
from tokenizer import StanfordCoreNlpTokenizer
import random

import spacy
nlp = spacy.load('en_core_web_sm')

# Add neural coref to SpaCy's pipe
import neuralcoref
neuralcoref.add_to_pipe(nlp)

config={"wiki":False,"weights":False,"eval_dataset":"KDD","weights_dataset":"", "verbose":False, "n":[10]}

labels = []
conv = {'-LRB-':'(',
        '-RRB-':')',
        '-LCB-':'{',
        '-RCB-':'}',
        '-LSB-':'[',
        '-RSB-':']'}

ake_path="C:/Users/dallal/Documents/GitHub/ake-datasets/" #https://github.com/dallal9/ake-datasets/

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
    with codecs.open(f, 'r', 'cp1252') as f:

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

def get_dataset_labels(dataset_name):
    if dataset_name == "KDD":
        js_file =  ake_path + "datasets/KDD/references/test.author.json"
    elif  dataset_name == "WWW":
        js_file =  ake_path + "datasets/WWW/references/test.author.json"
    elif dataset_name == "Inspec":
        js_file = ake_path + "datasets/Inspec/references/test.contr.json"
    elif dataset_name == "NUS":
        js_file = ake_path + "datasets/NUS/references/test.author.json"
    
    with open(js_file, 'r',encoding='utf-8') as f:
        keyphrases_gold = json.load(f)
    
    return keyphrases_gold

def get_text (dataset_name,label):
    if dataset_name == "KDD":
        f =  ake_path + "datasets/KDD/src/kpdata/KDD/abstracts/"+str(label)
    elif  dataset_name == "WWW":
        f =  ake_path + "datasets/WWW/src/kpdata/WWW/abstracts/"+str(label)
    elif dataset_name == "Inspec":
        f = ake_path + "datasets/Inspec/src/Hulth2003/"+str(label)+".abstr"
    elif dataset_name == "NUS":
        f = ake_path + "/datasets/NUS/src/data/'+str(label)"+"/"+str(label)+".txt"
    
    text = extract(f)  

    return text

keyphrases_gold = get_dataset_labels(config["eval_dataset"])

labels = list(keyphrases_gold.keys())

#random.shuffle(labels)


if config ["weights_dataset"] and config["weights"]:  
    keyphrases_W = get_dataset_labels(config["weights_dataset"])

    WS=[]
    for key in list(keyphrases_W.keys()):
        for k in keyphrases_W[key]:
            WS.append(k[0].lower())
    words={}
    for word in WS:
        try:
            words[word]+=1
        except:
            words[word]=1

    max_key = max(words, key=words.get)
    factor = words[max_key]

    for k in words:
        words[k] = words[k]/factor

for n in config["n"]: 
    P,R,F1=0.0,0.0,0.0
    for indx in tqdm(range(len(labels))):

        extractor = pke.unsupervised.PositionRank()
        label = labels[indx]
        text = get_text(config["eval_dataset"],label)
        #text[0]+=" "
        
        #text = " ".join(text)
        
        tokenizer = StanfordCoreNlpTokenizer("http://localhost", port = 9000)
  
        keyphrases_p=position_rank(text, tokenizer,alpha=0.85, window_size=2,num_keyphrase=n,weights=None,wiki=True)
    
        keyphrases_gold_p=[]
        for l in keyphrases_gold[label]:
            keyphrases_gold_p+=l

        keyphrases_p=list(set(keyphrases_p))
        p,r,f1=metric(keyphrases_gold_p,keyphrases_p)

        P+=p
        R+=r
        F1+=f1
        if indx%50==0 and config["verbose"]:
            print(P/(indx+1))
            print(R/(indx+1))
            print(F1/(indx+1))

    print(n,P/len(labels),R/len(labels),F1/len(labels))
