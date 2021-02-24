import bibtexparser
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from langdetect import detect

import re

from get_abstract import Article

def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

def append_to_parquet_table( dataframe, filepath=None, writer=None):
    """Method writes/append dataframes in parquet format.

    This method is used to write pandas DataFrame as pyarrow Table in parquet format. If the methods is invoked
    with writer, it appends dataframe to the already written pyarrow table.

    :param dataframe: pd.DataFrame to be written in parquet format.
    :param filepath: target file location for parquet file.
    :param writer: ParquetWriter object to write pyarrow tables in parquet format.
    :return: ParquetWriter object. This can be passed in the subsequenct method calls to append DataFrame
        in the pyarrow Table
    """
    table = pa.Table.from_pandas(dataframe)
    if writer is None:
        #print(table.schema)
        writer = pq.ParquetWriter(filepath, table.schema)
    writer.write_table(table=table)
    return writer







labels = ["author","journal","editor","volume","number","pages","month","year","doi", "pages","booktitle","publisher","series","comments" ,"notes","institution","bibsource"]

parser = bibtexparser.bparser.BibTexParser(common_strings=True)

bib_files= open("names.txt").read().splitlines()
err = open("error_name.txt","w")


outputFilePath = "bib_tug_dataset.parquet"
writer = None

titles = []
for bib_file in bib_files:
    print(len(titles))
    if len(titles)>5000:
        break
    try:
        with open("bibs/"+bib_file,encoding="utf-8") as bibtex_file:
            bibtex_str = bibtex_file.read()

        bib_database = bibtexparser.loads(bibtex_str, parser=parser)
        print(bib_file)
    except:
        err.write(bib_file+"\n")
        err.flush()
        continue


    #pretty(bib_database.entries[0])



    for entry in bib_database.entries:
        f = True

        '''Filtering entry by type'''
        if entry["ENTRYTYPE"] in ["inproceedings","article"]:
            parsed_entry = {"bib_file": bib_file}
            try:
                parsed_entry["keywords"] = str(entry["keywords"])
            except:
                continue

            try:
                parsed_entry["title"] = str(entry["title"])
                if parsed_entry["title"] in titles:
                    continue
                if detect(parsed_entry["title"]) not in ["en"]:
                    continue
            except:
                continue

            try:
                parsed_entry["abstract"] = str(entry["abstract"])
            except:
                try:
                    1/0
                    paper = Article(title=parsed_entry["title"])
                    abstract = paper.get_abstract()
                    if len(abstract)>100:
                        parsed_entry["abstract"] = abstract
                    else:
                        continue
                except:
                    continue

            for label in labels:
                try:
                    parsed_entry[label] = str(entry[label])
                except:
                    parsed_entry[label] = ""
            try:
                parsed_entry["url"] = str(entry["url"])
            except:
                try:
                    parsed_entry["url"] = str(entry["URL"])
                except:
                    parsed_entry["url"] = ""

            try:
                if int(parsed_entry["year"])<1970:
                    continue
            except:
                pass

            if f:

                titles.append(parsed_entry["title"])
                df = pd.DataFrame([parsed_entry,])

                if not df.empty:
                                writer = append_to_parquet_table(
                                    df, outputFilePath, writer)

if writer:
    writer.close()


