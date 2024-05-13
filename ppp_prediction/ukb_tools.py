import subprocess
from io import StringIO
from functools import reduce

import requests
from bs4 import BeautifulSoup
from multiprocessing import Pool
import pandas as pd
import numpy as np
import os
import yaml
from tqdm.notebook import tqdm



def extract_ukb_fields(fields,data="ukb676977.trans.tsv.gz", threads=4):
    """
    Extract UK Biobank fields from the UK Biobank data.
    """
    if isinstance(fields, dict):
        fields_dict = fields
        fields = [i for i in fields.keys()]

    else:
        fields_dict = None

    if isinstance(fields, str):
        fields = [fields]
    fields = [str(i) for i in fields]

    if len(fields) == 1:
    # if isinstance(fields, list):
        field = fields[0]
        cmd = f"tabix -h {data} {field} | datamash transpose | tail -n +2"
        print(cmd)
        result = subprocess.run(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if result.returncode != 0:
            # print(cmd)
            raise Exception(result.stderr)
        stdout = result.stdout
        if stdout == "":
            print(f"no data found in {field}")
            return None

        # first_line = stdout.split("\n")[0]
        # if all([isinstance(i, str) for i in first_line.split("\t")]):
        #     header = 0
        # else:
        #     header = None

        df = pd.read_csv(
            StringIO(stdout),
            sep="\t",
            #    header=header
        ).set_index("eid")
        return df

    else:
        threads = min(threads, len(fields))
        print(threads)
        with Pool(threads) as p:
            # field_names = list(tqdm(p.imap(get_ukb_field_name, id), total=len(id)))
            extract_list = list(
                tqdm(p.imap(extract_ukb_fields, fields), total=len(fields))
            )

        df = reduce(
            lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how="outer"),
            extract_list,
        )

    new_columns_map = {}
    for col in df.columns:
        code = col.split("-")[0]
        new_name = "f" + col

        if fields_dict is not None:
            filedsName = fields_dict.get(code, None)
            if filedsName is not None:
                new_name = filedsName + "_" + new_name

        new_columns_map[col] = new_name.replace(" ", "_").replace(".", "_").replace("-", "_")

        # if fields_dict is not None:
        #     new_columns_map = {}
        #     for col in df.columns:
        #         split_id = col.split("-")
        #         split_id[0] = fields_dict.get(split_id[0], split_id[0])
        #         new_columns_map[col] = "_".join(split_id).replace(".", "_")

        df = df.rename(columns=new_columns_map)
    return df


def get_ukb_field_name(id, threads=4):
    if isinstance(id, list):
        threads = min(threads, len(id))

        with Pool(threads) as p:
            field_names = list(tqdm(p.imap(get_ukb_field_name, id), total=len(id)))
        return field_names

    url = f"https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id={id}"  # 替换为你要爬取的页面的URL
    response = requests.get(url)
    if response.status_code == 200:
        page = response.text
        soup = BeautifulSoup(page, "html.parser")
        fieldname = (
            soup.find("table", {"summary": "Identification"})
            .select("tr")[0]
            .select("td")[1]
            .text
        )  # find the field name
        fieldname = fieldname.replace(" ", "_")
        return fieldname
    else:
        return None


from typing import Dict


def define_disease(
    data,
    disease_code_dict: Dict = None,
    ph: str = None,
    codes: list = None,
    time_col="icd10_date",
    code_col="icd10_code",
):
    if ph is not None and codes is not None:

        tmp = data[["eid"]].drop_duplicates()
        matched = data[data[code_col].str.contains("|".join(codes))]
        matched = (
            matched.sort_values(time_col)
            .groupby("eid")
            .apply(lambda x: x.head(1))
            .reset_index(drop=True)
            .assign(pheno=1, date=lambda x: x[time_col])
            .rename(columns={"date": f"{ph}_event_date", "pheno": f"{ph}_event"})
        )
        tmp_ph = tmp.merge(matched, how="left", on="eid")
        tmp_ph[f"{ph}_event"] = tmp_ph[f"{ph}_event"].fillna(0)
        tmp_ph = tmp_ph.drop(columns=[time_col, code_col])

        return tmp_ph
    else:

        disease_list = [
            define_disease(
                data=data, ph=ph, codes=codes, time_col=time_col, code_col=code_col
            )
            for ph, codes in tqdm(disease_code_dict.items(), desc="Define disease")
        ]
        return reduce(lambda x, y: pd.merge(x, y, on="eid", how="outer"), disease_list)