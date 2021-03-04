import bibtexparser
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from langdetect import detect

import re

from get_abstract import Article

import logging

logging.getLogger("bibtexparser").setLevel(logging.WARNING)


def append_to_parquet_table(dataframe, filepath=None, writer=None):
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
        writer = pq.ParquetWriter(filepath, table.schema)
    writer.write_table(table=table)
    return writer


labels = ["fullpaper","author", "journal", "editor", "volume", "number", "pages", "month", "year", "doi", "pages", "booktitle",
          "publisher", "series", "comments", "notes", "institution", "bibsource"]

bib_list = "names.txt"
failed_files = "error_name.txt"
outputFilePath = "bib_tug_dataset_full.parquet"

max_entries = 30000
extract_abstract = False
skip_total = 25
skip_false = 0.4

bib_files = open(bib_list).read().splitlines()
err = open(failed_files, 'w')

writer = None

titles = []

for bib_file in bib_files:
    parser = bibtexparser.bparser.BibTexParser(common_strings=True)
    print("Total entries: " , len(titles))
    print("Processing ",bib_file)
    entries = []
    count = 0

    if len(titles) > max_entries:
        break

    try:
        with open("bibs/" + bib_file, encoding="utf-8") as bibtex_file:
            bibtex_str = bibtex_file.read()
    except:
        err.write(bib_file + "\n")
        err.flush()
        continue

    bibtex_str = re.sub(r'  acknowledgement =.+\n', '', bibtex_str)

    entry = []
    Entries = []
    write = False

    flag = False
    for line in bibtex_str.splitlines():
        entry.append(line)

        for each in ["@article", "@Article", "@inproceedings", "@Inproceedings"]:
            if each in line:
                if not flag:
                    entry = "\n".join(entry[:-1])
                    Entries.append(entry)
                    entry = [line]
                    flag = True

        if line and flag:
            if line[0] == "}" and len(entry) > 1:
                entry = "\n".join(entry)
                if "keywords = " in entry:
                    if "abstract" in entry:
                        Entries.append(entry)
                entry = []

    bibtex_str = "\n\n".join(Entries)

    try:
        bib_database = bibtexparser.loads(bibtex_str, parser=parser)
    except:
        err.write(bib_file + "\n")
        err.flush()
        continue

    for entry in bib_database.entries:

        '''Filtering entry by type'''
        if entry["ENTRYTYPE"].lower() in ["inproceedings", "article"]:
            parsed_entry = {"bib_file": bib_file}
            try:
                parsed_entry["keywords"] = str(entry["keywords"])
                if len(parsed_entry["keywords"]) < 50:
                    continue
            except:
                continue

            try:
                parsed_entry["title"] = str(entry["title"])
                parsed_entry["title"] = parsed_entry["title"].replace("{", "")
                parsed_entry["title"] = parsed_entry["title"].replace("}", "")
                if parsed_entry["title"] in titles:
                    continue
                if detect(parsed_entry["title"]) not in ["en"]:
                    continue
            except:
                continue

            try:
                parsed_entry["abstract"] = str(entry["abstract"])
                if len(parsed_entry["abstract"]) < 50:
                    continue

            except:
                try:
                    if extract_abstract:
                        if count < skip_total:
                            paper = Article(title=parsed_entry["title"])
                            abstract = paper.get_abstract()
                            if len(abstract) > 500:
                                parsed_entry["abstract"] = abstract

                            else:
                                count += skip_false
                                continue
                        else:
                            count += skip_false
                            continue
                    else:
                        continue
                except:
                    count += skip_false
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
                if int(parsed_entry["year"]) < 1900:
                    continue
            except:
                pass

            count += 1
            titles.append(parsed_entry["title"])
            entries.append(parsed_entry)

    df = pd.DataFrame(entries)

    if not df.empty:
        writer = append_to_parquet_table(df, outputFilePath, writer)

if writer:
    writer.close()
