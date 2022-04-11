#!/usr/bin/env python3
import pandas
import numpy as np
import torch
from pyarrow import parquet
import os


def read_from_parquet(small_set=False):
    print("reading parquet files")
    initial_dir = os.getcwd()
    home_dir = os.environ.get("HOME")
    # os.chdir(home_dir + "/gout-data")
    # print("reading '" + home_dir + "/gout-data/geno_10k.parquet")
    # geno_tmp = np.matrix([[1,1],[1,0]])
    geno_tmp = parquet.read_table(home_dir + "/gout-data/gwas_associated_only/geno_10k.parquet")
    urate_tmp = parquet.read_table(home_dir + "/gout-data/gwas_associated_only/urate_10k.parquet")
    urate = urate_tmp.to_pandas()
    geno = geno_tmp.to_pandas()
    geno = np.transpose(np.matrix(geno_tmp))
    print("geno mat shape: ", geno.shape)
    print("urate shape: ", urate.shape)

    # parquet files are already filtered
    # withdrawn_ids = pandas.read_csv("w12611_20220222.csv") usable_ids = list(
    #     (set(urate_pd.eid).intersection(geno_tmp.index))
    #     - set(withdrawn_ids.withdrawn_ids)
    # )

    # urate = urate_tmp[urate_tmp["eid"].isin(usable_ids)].sort_values("eid")
    # geno = geno_tmp.filter(usable_ids, axis="rows").sort_index()
    if small_set:
        num_samples = 1000
        num_snps = 200
        geno = geno[0:num_samples, 0:num_snps]
        urate = urate[0:num_samples]

    # if sum(urate.eid != geno.index) > 0:
    #     raise Exception("genotypy indeices do not match urate indices")

    geno_mat = np.matrix(geno)
    num_zeros = np.sum(geno_mat == 0)
    num_ones = np.sum(geno_mat == 1)
    num_twos = np.sum(geno_mat == 2)
    num_unknown = np.sum(geno_mat == -9)
    num_non_zeros = np.sum(geno_mat != 0)
    num_nan = np.sum(np.isnan(geno_mat))
    total_num = num_zeros + num_non_zeros
    # geno_med = np.bincount(geno_mat)
    # values, counts = np.unique(geno_mat, return_counts=True)
    values, counts = np.unique(np.matrix.flatten(geno_mat), axis=1, return_counts=True)

    most_common = values[0, np.argmax(counts)]

    print(
        "geno mat contains {:.2f}%% zeros, {:.2f}%% ones, {:.2f}%% twos {:.2f}%% nans {:.2f}%% unknowns (-9)".format(
            100.0 * num_zeros / total_num,
            100 * (num_ones / total_num),
            100 * (num_twos / total_num),
            100.0 * num_nan / total_num,
            100.0 * num_unknown / total_num,
        )
    )
    print("{:.3f}%% has gout".format(100 * np.sum(urate.gout) / len(urate)))

    # os.chdir(initial_dir)
    return geno_mat, urate


def read_from_csv(small_set=False):
    geno_tmp = pandas.read_csv("geno_10k.csv.gz", delimiter=";")
    urate_tmp = pandas.read_csv("urate.csv")
    withdrawn_ids = pandas.read_csv("w12611_20220222.csv")
    usable_ids = list(
        (set(urate_tmp.eid).intersection(geno_tmp.index))
        - set(withdrawn_ids.withdrawn_ids)
    )

    urate = urate_tmp[urate_tmp["eid"].isin(usable_ids)].sort_values("eid")
    geno = geno_tmp.filter(usable_ids, axis="rows").sort_index()
    if small_set:
        num_samples = 1000
        num_snps = 200
        geno = geno[0:num_samples, 0:num_snps]
        urate = urate[0:num_samples]

    if sum(urate.eid != geno.index) > 0:
        raise Exception("genotypy indeices do not match urate indices")

    geno_mat = np.matrix(geno)
    num_zeros = np.sum(geno_mat == 0)
    num_ones = np.sum(geno_mat == 1)
    num_twos = np.sum(geno_mat == 2)
    num_unknown = np.sum(geno_mat == -9)
    num_non_zeros = np.sum(geno_mat != 0)
    num_nan = np.sum(np.isnan(geno_mat))
    total_num = num_zeros + num_non_zeros
    # geno_med = np.bincount(geno_mat)
    values, counts = np.unique(geno_mat, return_counts=True)
    most_common = values[np.argmax(counts)]

    print(
        "geno mat contains {:.2f}%% zeros, {:.2f}%% ones, {:.2f}%% twos {:.2f}%% nans {:.2f}%% unknowns (-9)".format(
            100.0 * num_zeros / total_num,
            100 * (num_ones / total_num),
            100 * (num_twos / total_num),
            100.0 * num_nan / total_num,
            100.0 * num_unknown / total_num,
        )
    )
    print("{:.3f}%% has gout".format(100 * np.sum(urate.gout) / len(urate)))

    return geno_mat, urate


def read_from_plink(remove_nan=True, small_set=False):
    geno_tmp = read_plink1_bin("chr1_tmp.bed", "chr1_tmp.bim", "chr1_tmp.fam")
    geno_tmp["sample"] = pandas.to_numeric(geno_tmp["sample"])
    urate_tmp = pandas.read_csv("urate.csv")
    withdrawn_ids = pandas.read_csv("w12611_20220222.csv")

    usable_ids = list(set(urate_tmp.eid) - set(withdrawn_ids.withdrawn_ids))
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
        "geno mat contains {:.2f}%% zeros, {:.2f}%% ones, {:.2f}%% twos {:.2f}%% nans".format(
            100.0 * num_zeros / total_num,
            100 * (num_ones / total_num),
            100 * (num_twos / total_num),
            100.0 * num_nan / total_num,
        )
    )
    print("{}%% has gout".format(100 * np.sum(urate.gout) / len(urate)))

    if remove_nan:
        geno_mat[np.isnan(geno_mat)] = most_common

    return geno_mat, urate


if __name__ == "__main__":
    geno, pheno = read_from_parquet()
    print(geno.shape)
    print(pheno.shape)
