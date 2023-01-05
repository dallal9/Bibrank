import csv
import json
import random
import sys
import time
from typing import List

import pandas as pd
from tqdm import tqdm

from Datasets.utils import clean, read_parquet

sys.path.append("Models/")
from bibrank import get_weights

from Models import *


def mask(text1: str, text2: str) -> List[str]:
    """
    a simple vectorization function
    """
    base = 0
    vectors = {}
    vector1, vector2 = [], []
    for phrase in text1:
        if phrase not in vectors:
            vectors[phrase] = base
            base += 1
        vector1.append(vectors[phrase])

    for phrase in text2:
        if phrase not in vectors:
            vectors[phrase] = base
            base += 1
        vector2.append(vectors[phrase])
    return vector1, vector2


def get_recall(l1: List[str], l2: List[str]) -> float:
    occur_count = 0.0

    for each in l2:
        if each in l1:
            occur_count += 1.0
    return occur_count / float(len(l1))


def get_precision(l1: List[str], l2: List[str]) -> float:
    occur_count = 0.0

    for each in l2:
        if each in l1:
            occur_count += 1.0
    return occur_count / float(len(l2))


def get_f1(r: float, p: float) -> float:
    if r + p == 0:
        return 0.0
    return (2.0 * p * r) / (p + r)


def get_Rprecision(phrase1: List[str], phrase2: List[str]) -> float:
    """
    relaxed since it uses "in" instead of checkign the position of words
    """
    occur_count = 0.0

    for w1 in phrase2.split():
        if w1 in phrase1:
            occur_count += 1

    return occur_count / max(len(phrase2.split()), len(phrase1.split()))


def get_all_Rprecision(
    gold_keywords: List[str], predicted_keywords: List[str]
) -> float:
    scores = []
    for gphrase in gold_keywords:
        rscores = []
        for pphrase in predicted_keywords:
            rscores.append(get_Rprecision(gphrase, pphrase))
        scores.append(max(rscores))

    return sum(scores) / len(scores)


def get_scores(gold_keywords: List[str], predicted_keywords: List[str]) -> List[float]:
    recall = get_recall(gold_keywords, predicted_keywords)
    precision = get_precision(gold_keywords, predicted_keywords)
    f1 = get_f1(recall, precision)
    Rprecision = get_all_Rprecision(gold_keywords, predicted_keywords)
    return f1, recall, precision, Rprecision


def get_all_scores(
    gold_keywords: List[str],
    predicted_keywords: List[str],
    text: str = "",
    adjust: bool = False,
) -> List:
    """
    SemEval-2010 Task 5, micro averaged f1, recall, precision
    """
    metrics = get_scores(gold_keywords, predicted_keywords)

    adjusted_gold = []
    adjusted_metrics = []
    if adjust:
        if text:
            for each in gold_keywords:
                if each in text:
                    adjusted_gold.append(each)
            if len(adjusted_gold) < 1:
                adjusted_metrics = []
            else:
                adjusted_metrics = get_scores(adjusted_gold, predicted_keywords)

    return metrics, adjusted_metrics


def get_key_abs(
    filepath: str,
    year1: int = 1900,
    year2: int = 2020,
    bib_files: List[str] = [],
    types: List[str] = [],
    journals: List[str] = [],
    count: int = None,
    rand: bool = False,
):

    jfile = open("Datasets/bib_info.json").read()
    tables, names = json.loads(jfile)
    new_tables = {}
    for table in tables:
        new_tables[table] = []
        for f in tables[table]:
            new_tables[table].append(list(f.keys())[0])

    df = read_parquet(filepath)

    if bib_files:
        try:
            df = df[df["bib_file"].isin(bib_files)]
        except:
            df = df[df["bibsource"].isin(bib_files)]

    if types:
        types_list = []
        for type in types:
            types_list.extend(new_tables[type])
        df = df[df["bib_file"].isin(types_list)]

    if journals:
        df = df[df["journal"].isin(journals)]

    # year
    try:
        df["year"] = df["year"].astype("int")
        df = df[df["year"] >= year1]
        df = df[df["year"] <= year2]
    except:
        pass

    if rand and count:
        df = df.sample(n=count)
    else:
        df = df[:count]

    keywords = df["keywords"].tolist()
    abstracts = df["abstract"].tolist()
    c1, c2 = 0.0, 0.0

    processed_keywords = []
    for i in range(len(keywords)):
        if ";" in keywords[i]:
            keyword = keywords[i].split(";")
        elif "," in keywords[i]:
            keyword = keywords[i].split(",")
        elif "\t" in keywords[i]:
            keyword = keywords[i].split("\t")
        else:
            keyword = keywords[i].split(" to")

        new = []
        for key in keyword:
            if "---" in key:
                key = key.split("---")
                new.extend([clean(k) for k in key])
            else:
                new.append(clean(key))
        processed_keywords.append(new)

    return processed_keywords, abstracts


def eval_file(
    filepath,
    model,
    year1=1900,
    year2=2020,
    bib_files=[],
    types=[],
    journals=[],
    limit=None,
    rand=False,
    log=True,
    model_param="",
    outputpaths=["output.json", "output.tsv"],
    bib_weights={},
    top_n=10,
):

    t1 = time.time()

    keywords_gold, abstracts = get_key_abs(
        filepath, year1, year2, bib_files, types, journals, limit, rand
    )

    if bib_weights:
        keywords_gold_w, t = get_key_abs(
            filepath=bib_weights["dataset"],
            year1=bib_weights["year1"],
            year2=bib_weights["year2"],
            types=bib_weights["types"],
        )

        weights = get_weights(keywords_gold_w)

        model = BibRank(weights)

    all_scores = []
    all_scores_adjust = []
    T0 = 0.0
    T1 = 0.0
    data_file_ = open(model.model_name.lower() + ".tsv", "w")
    for i in tqdm(range(len(keywords_gold))):

        t3 = time.time()

        predicted_keywords = model.get_keywords(abstracts[i], n=top_n)

        t4 = time.time()
        try:
            scores = get_all_scores(
                keywords_gold[i], predicted_keywords[0], abstracts[i], adjust=True
            )

        except:
            scores = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]

        data_line = [abstracts[i], keywords_gold[i], predicted_keywords[0]]
        data_line.extend(scores)
        data_line = [str(i) for i in data_line]
        data_line = "\t".join(data_line)
        data_file_.write(data_line)
        data_file_.write("\n")

        t5 = time.time()
        all_scores.append(scores[0])
        if scores[1]:
            all_scores_adjust.append(scores[1])
        T0 += t4 - t3
        T1 += t5 - t4
    all_scores = pd.DataFrame(all_scores)
    all_scores_adjust = pd.DataFrame(all_scores_adjust)

    counts = [len(all_scores), len(all_scores_adjust)]

    all_scores = list(all_scores.mean(axis=0))
    all_scores_adjust = list(all_scores_adjust.mean(axis=0))
    data_line = list(all_scores)
    data_line.extend(all_scores_adjust)
    data_line = [str(i) for i in data_line]
    data_line = "\t".join(data_line)
    data_file_.write(data_line)
    data_file_.write("\n")

    t = time.time() - t1
    label = "".join(
        random.choice("1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM")
        for x in range(8)
    )

    output = {
        "label": label,
        "file_path": filepath,
        "year1": year1,
        "year2": year2,
        "bib_files": bib_files,
        "types": types,
        "journals": journals,
        "limit": limit,
        "model_name": model.model_name,
        "model_param": model_param,
        "counts": counts,
        "scores": all_scores,
        "scores_adjusted": all_scores_adjust,
        "time": t,
        "random": rand,
    }
    if log:
        f1 = open(outputpaths[0], "a")
        json.dump(output, f1)
        f1.write("\n")

        with open(outputpaths[1], "a", newline="") as out_file:
            tsv_writer = csv.writer(out_file, delimiter="\t")
            tsv_writer.writerow(
                [
                    label,
                    model.model_name,
                    filepath,
                    str(counts[0]),
                    str(counts[1]),
                    str(all_scores[0]),
                    str(all_scores[1]),
                    str(all_scores[2]),
                    str(all_scores[3]),
                    str(all_scores_adjust[0]),
                    str(all_scores_adjust[1]),
                    str(all_scores_adjust[2]),
                    str(all_scores_adjust[3]),
                ]
            )

    return all_scores, all_scores_adjust
