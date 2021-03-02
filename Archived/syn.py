import spacy
import json
from tqdm import tqdm
import codecs
import csv

conv = {'-LRB-':'(',
        '-RRB-':')',
        '-LCB-':'{',
        '-RCB-':'}',
        '-LSB-':'[',
        '-RSB-':']'}

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
nlp = spacy.load("en_core_web_sm")

with open('C:/Users/dallal/Documents/GitHub/ake-datasets/datasets/KDD/references/test.author.json', 'r',encoding='utf-8') as f:
    keyphrases_gold = json.load(f)
keyphrases = {}

for key in list(keyphrases_gold.keys()):
    keyphrases[key]=[]
    for k in keyphrases_gold[key]:
        keyphrases[key].append(k[0].lower())
fil=open("out.csv","w",encoding="utf-8")
labels = list(keyphrases_gold.keys())
out=[]
for indx in tqdm(range(len(labels))):

        
        label = labels[indx]
        f='C:/Users/dallal/Documents/GitHub/ake-datasets/datasets/NUS/test/'+str(label)+'.xml'
        f= 'C:/Users/dallal/Documents/GitHub/ake-datasets/datasets/NUS/src/kpdata/NUS/abstracts/'+str(label)
        f= 'C:/Users/dallal/Documents/GitHub/ake-datasets/datasets/NUS/src/data/'+str(label)+'/'+str(label)+'.txt'
        f= 'C:/Users/dallal/Documents/GitHub/ake-datasets/datasets/KDD/src/kpdata/KDD/abstracts/'+str(label)
        text = extract(f)
        #text[0]+=" "
        #print(keyphrases[label])
        
        text = " ".join(text)
        text = text.lower()
        doc = nlp(text)

        span = doc[doc[2].left_edge.i : doc[2].right_edge.i+1]
    
            
        fil.write(text+"\n")
        fil.write("\n")
        fil.write(str(keyphrases[label])+"\n")
        fil.write("\n")
        with doc.retokenize() as retokenizer:
            retokenizer.merge(span)
        for token in doc:
            #print(token.text)
            if token.text in keyphrases[label]:
                #print(keyphrases[label])
            
                out.append([token.text, token.pos_, token.dep_, token.head.text])
                fil.write(token.text+" "+ token.pos_+" "+ token.dep_+" "+ token.head.text)
                fil.write("\n")
                #print(token.pos_)
        fil.write("\n")
pos={}
dep={}
for each in out:
    try:
        pos[each[1]]+=1
    except:
        pos[each[1]]=1
    
    try:
        dep[each[2]]+=1
    except:
        dep[each[2]]=1

print(pos) 
print(dep)


