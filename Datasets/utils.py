import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import re



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
        try:
            old_table = pq.read_table(filepath)
            writer = pq.ParquetWriter(filepath, old_table.schema)
            writer.write_table(table=old_table)
        except:
            writer = pq.ParquetWriter(filepath, table.schema)
    writer.write_table(table=table)
    return writer


def read_parquet (filepath):
    table = pq.read_table(filepath)
    df = table.to_pandas()
    return df


labels = ["fullpaper","author", "journal", "editor", "volume", "number", "pages", "month", "year", "doi", "pages", "booktitle",
          "publisher", "series", "comments", "notes", "institution", "bibsource"]

from bs4 import BeautifulSoup
import re

def get_bib_info (file="bib.html"):
    s = open(file).read()

    soup = BeautifulSoup(s, "lxml")

    gdp_table = soup.find_all("table",attrs={"width":750})
    tables = {}
    Names = {}
    for table in gdp_table:
        As= table.find_all("a")
        for a in As:
            if a.has_attr("name"):
                tables[a["name"]] = []
                text = re.sub(' +', ' ',a.text.strip().replace('\n', ' '))
                Names[a["name"]] = text
                label = a["name"]

            elif a.has_attr("href"):
                text = re.sub(' +', ' ',a.text.strip().replace('\n', ' '))
                tables[label].append({a["href"].split("/")[-1]:text})


    return tables,Names


def clean (text):
    text = text.strip()
    text = text.replace("\n", " ")
    text = re.sub(r" +"," ",text)

    return text.lower()



