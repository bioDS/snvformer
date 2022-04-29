#!/usr/bin/env python3
import os
import pandas
import numpy as np
from pandas_plink import read_plink1_bin
from pyarrow import parquet
import torch
from process_snv_mat import get_tok_mat

data_dir = os.environ['UKBB_DATA'] + "/"
# gwas_dir = os.environ['UKBB_DATA'] + "/gwas_associated_only/"
gwas_dir = os.environ['UKBB_DATA'] + "/"
plink_base = os.environ['PLINK_FILE']
urate_file = os.environ['URATE_FILE']

class Tokenised_SNVs:
    def __init__(self, geno, use_ai_encoding):
        # val == 0 means we have two 'a0' vals
        # val == 2 means two 'a1' vals
        # val == 1 means one of each
        tok_mat, tok_to_string, string_to_tok, num_toks = get_tok_mat(geno, use_ai_encoding)

        self.string_to_tok = string_to_tok
        self.tok_to_string = tok_to_string
        self.tok_mat = tok_mat
        self.num_toks = num_toks
        self.positions = torch.tensor(geno.pos.values, dtype=torch.long)

def read_from_plink(remove_nan=False, small_set=False, subsample_control=True, use_ai_encoding=False):
    print("using data from:", data_dir)
    bed_file = gwas_dir+plink_base+".bed"
    bim_file = gwas_dir+plink_base+".bim"
    fam_file = gwas_dir+plink_base+".fam"
    print("bed_file:", bed_file)
    geno_tmp = read_plink1_bin(bed_file, bim_file, fam_file)
    geno_tmp["sample"] = pandas.to_numeric(geno_tmp["sample"])
    urate_tmp = pandas.read_csv(data_dir + urate_file)
    withdrawn_ids = pandas.read_csv(data_dir + "w12611_20220222.csv", header=None, names=["ids"])

    usable_ids = list(set(urate_tmp.eid) - set(withdrawn_ids.ids))
    urate = urate_tmp[urate_tmp["eid"].isin(usable_ids)]
    del urate_tmp
    geno = geno_tmp[geno_tmp["sample"].isin(usable_ids)]
    del geno_tmp
    if small_set:
        num_samples = 1000
        num_snps = 200
        geno = geno[0:num_samples, 0:num_snps]
        urate = urate[0:num_samples]

    if (subsample_control):
        gout_cases = urate[urate.gout]["eid"]
        non_gout_cases = urate[urate.gout == False]["eid"]
        # non_gout_cases = np.where(urate.gout == False)[0]
        non_gout_sample = np.random.choice(non_gout_cases, size=len(gout_cases), replace=False)
        sample_ids = list(set(gout_cases).union(non_gout_sample))
        urate = urate[urate["eid"].isin(sample_ids)]
        geno = geno[geno["sample"].isin(sample_ids)]

    geno_mat = geno.values
    positions = np.asarray(geno.pos)

    num_zeros = np.sum(geno_mat == 0)
    num_ones = np.sum(geno_mat == 1)
    num_twos = np.sum(geno_mat == 2)
    num_non_zeros = np.sum(geno_mat != 0)
    num_nan = np.sum(np.isnan(geno_mat))
    total_num = num_zeros + num_non_zeros
    # geno_med = np.bincount(geno_mat)
    values, counts = np.unique(geno_mat, return_counts=True)
    most_common = values[np.argmax(counts)]

    print(
        "geno mat contains {:.2f}% zeros, {:.2f}% ones, {:.2f}% twos {:.2f}% nans".format(
            100.0 * num_zeros / total_num,
            100 * (num_ones / total_num),
            100 * (num_twos / total_num),
            100.0 * num_nan / total_num,
        )
    )
    print("{:.2f}% has gout".format(100 * np.sum(urate.gout) / len(urate)))

    if remove_nan:
        geno_mat[np.isnan(geno_mat)] = most_common

    # we ideally want the position and the complete change
    snv_toks = Tokenised_SNVs(geno, use_ai_encoding)

    return snv_toks, urate

if __name__ == "__main__":
    snv_toks, urate = read_from_plink()