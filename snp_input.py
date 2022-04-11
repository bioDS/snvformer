#!/usr/bin/env python3
import os
import pandas
import numpy as np
from pandas_plink import read_plink1_bin
from pyarrow import parquet

data_dir = os.environ['UKBB_DATA'] + "/"
gwas_dir = os.environ['UKBB_DATA'] + "/gwas_associated_only/"

def read_from_plink(remove_nan=True, small_set=False):
    print("using data from:", data_dir)
    bed_file = gwas_dir+"xaa_chr1_tmp.bed"
    bim_file = gwas_dir+"xaa_chr1_tmp.bim"
    fam_file = gwas_dir+"xaa_chr1_tmp.fam"
    print("bed_file:", bed_file)
    geno_tmp = read_plink1_bin(bed_file, bim_file, fam_file)
    geno_tmp["sample"] = pandas.to_numeric(geno_tmp["sample"])
    urate_tmp = pandas.read_csv(data_dir + "urate.csv")
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

    geno_mat = geno.values

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

    return geno_mat, urate

if __name__ == "__main__":
    geno_urate = read_from_plink()