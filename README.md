# BibRank: Automatic Keyphrase Extraction Platform Using Metadata

BibRank, a new semi-supervised automatic keyphrase extraction method that exploits an information-rich dataset collected by parsing bibliographic data in BibTeX format.

This repository contains BibRank implementation, Bib dataset, a platform to support datasets creation, different models processing, and running different evaluations. 


## Installation 

### 1. install all required dependencies. 

```
pip install -r requirements.txt
```
### 2. install extra dependencies for specific models. 

- Download and install dependencies for Embedrank

    Using their Github repo: https://github.com/AnzorGozalishvili/unsupervised_keyword_extraction


- BibRank and PositionRank use Standford CoreNLP Toolkit

    Download: https://stanfordnlp.github.io/CoreNLP/download.html

## Directory Structure
```

│   KeyEval.py
│   output.json
│   output.tsv
│   requirements.txt
│
├───Datasets
│   │   bib_info.json
│   │   error_name.txt
│   │   get_abstract.py
│   │   names.txt
│   │   ProcessBib.py
│   │   ProcessText.py
│   │   ProcessXML.py
│   │   utils.py
│   │
│   ├───DataFiles
│   │       bib_tug_dataset_full.parquet
│   │       text_dataset_full.parquet
│   │
└───Models
    │   Models.py
    │   position_rank.py
    │   test_position.py
    │   tokenizer.py
    
```