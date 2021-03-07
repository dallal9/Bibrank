from utils import append_to_parquet_table
from utils import labels as parsed_labels

import codecs
from tqdm import tqdm
import json
import pandas as pd


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

    return text


def extract_text(f):
    with codecs.open(f, 'r', 'cp1252') as f:
        text = f.read().splitlines()
    return  text

def get_keyphrases_gold(labels_json):
    with open(labels_json, 'r',encoding='utf-8') as f:
        keyphrases_gold = json.load(f)
    return keyphrases_gold


def get_label_weights(labels_json, factor):
    """
    used with the modified position rank model
    it can be used for any model which needs pre-defined weights for the labels

    """
    labels = get_labels(labels_json)
    words = []
    for key in list(keyphrases_W.keys()):
        for k in keyphrases_W[key]:
            words.append(k[0].lower())
    words_count = {}
    for word in words:
        try:
            words_count[word] += 1
        except:
            words_count[word] = 1

    max_key = max(words_count, key=words_count.get)
    factor = words_count[max_key]

    for k in words_count:
        words_count[k] = words_count[k] / factor

    return words_count


def process_dataset(labels_json, text_path, file_ext, dataset_name, text_type, split_type):
    """
    mainly designed for ake-datasets
    text_type: full text or abstract
    split_type: test or train
    """
    keyphrases_gold = get_keyphrases_gold(labels_json)
    labels = list(keyphrases_gold.keys())
    parsed_entries = []

    for indx in tqdm(range(len(labels))):
        parsed_entry = {}

        label = labels[indx]
        f = text_path + str(label) + ".txt"

        # abstract = extract(f)
        # parsed_entry["title"] = abstract[0]
        # parsed_entry["abstract"] = "\n".join(abstract[1:])


        text = extract(f)
        title = text[1]
        i,u=3,0
        for ii in range(len(text)):
            if "--B" in text[ii]:
                u=ii
                break
        abstract = "\n".join(text[i:u])
        paper = "\n".join(text[u+1:])



        parsed_entry["title"] = title
        parsed_entry["abstract"] = abstract
        parsed_entry["fullpaper"] = paper

        # text = extract_text(f)
        # parsed_entry["title"] = text[0]
        # text = "\n".join(text[1:])
        # l=text.split("\n\n")
        # parsed_entry["abstract"] =str(l[0])
        # parsed_entry["fullpaper"] = "\n\n".join(l[1:])

        keywords = [keyword[0] for keyword in keyphrases_gold[label]]
        parsed_entry["keywords"] = "\t".join(keywords)
        parsed_entry["bibsource"] = dataset_name
        parsed_entry["notes"] = split_type
        parsed_entry["url"] = ""

        for label in parsed_labels:
            try:
                parsed_entry[label]
            except:
                parsed_entry[label] = ""


        parsed_entries.append(parsed_entry)
    return parsed_entries


"""
parameters
"""
labels_json = "C:/dallal/MScTartu/Courses/spring21/Thesis/dataset/ake-datasets/datasets/ACM/references/test.author.json"
text_path = "C:/dallal/MScTartu/Courses/spring21/Thesis/dataset/ake-datasets/datasets/ACM/src/all/"
file_ext = ".txt"
dataset_name = "ACM"
text_type = "fullpaper"
split_type = "test"
parsed_entries = process_dataset(labels_json, text_path, file_ext, dataset_name, text_type, split_type)
outputFilePath = "text_dataset_full.parquet"

df = pd.DataFrame(parsed_entries)
writer = None
if not df.empty:
    writer = append_to_parquet_table(df, outputFilePath, writer)

if writer:
    writer.close()

print(parsed_entries[0])


