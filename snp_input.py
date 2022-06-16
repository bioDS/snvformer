#!/usr/bin/env python3
import os
import pandas
import numpy as np
from pandas_plink import read_plink1_bin
from pyarrow import parquet
import torch
from torch import tensor
from torch.utils import data
from process_snv_mat import get_tok_mat
import math
from os.path import exists
import pickle
import csv
from environ import *

# data_dir = os.environ['UKBB_DATA'] + "/"
# # gwas_dir = os.environ['UKBB_DATA'] + "/gwas_associated_only/"
# gwas_dir = os.environ['UKBB_DATA'] + "/"
# plink_base = os.environ['PLINK_FILE']
# pretrain_plink_base = os.environ['PRETRAIN_PLINK_FILE']
# urate_file = os.environ['URATE_FILE']
data_dir = ukbb_data
gwas_dir = ukbb_data
pretrain_plink_base = pretrain_base
urate_file = pheno_file
cache_dir = "./cache/"

test_ids_file   = cache_dir + "test_ids.csv"
train_ids_file  = cache_dir + "train_ids.csv"
verify_ids_file = cache_dir + "verify_ids.csv"

class Tokenised_SNVs:
    def __init__(self, geno, encoding: int):
        # val == 0 means we have two 'a0' vals
        # val == 2 means two 'a1' vals
        # val == 1 means one of each
        if (encoding <= 3):
            tok_mat, tok_to_string, string_to_tok, num_toks = get_tok_mat(geno, encoding)
        elif (encoding == 4):
            tok_mat, is_nonref_mat, alleles_differ_mat, diff_lens, tok_to_string, string_to_tok, num_toks = get_tok_mat(geno, encoding)
            self.is_nonref_mat = torch.tensor(is_nonref_mat, dtype=torch.bool)
            self.alleles_differ_mat = torch.tensor(alleles_differ_mat, dtype=torch.bool)
            self.diff_lens_mat = torch.tensor(is_nonref_mat, dtype=torch.int32)

        self.snp_ids = geno.snp.values
        self.ids = geno.iid.values
        self.string_to_tok = string_to_tok
        self.tok_to_string = tok_to_string
        self.tok_mat = tok_mat
        self.num_toks = num_toks
        self.positions = torch.tensor(geno.pos.values, dtype=torch.long)

def read_from_plink(remove_nan=False, subsample_control=True, encoding: int = 2, test_frac=0.3, verify_frac=0.05, summarise_genos=False):
    print("using data from:", data_dir)
    bed_file = gwas_dir+plink_base+".bed"
    bim_file = gwas_dir+plink_base+".bim"
    fam_file = gwas_dir+plink_base+".fam"
    print("bed_file:", bed_file)
    geno_tmp = read_plink1_bin(bed_file, bim_file, fam_file)
    geno_tmp["sample"] = pandas.to_numeric(geno_tmp["sample"])
    urate_tmp = pandas.read_csv(data_dir + urate_file)
    withdrawn_ids = pandas.read_csv(data_dir + "w12611_20220222.csv", header=None, names=["ids"])

    print("removing withdrawn ids")
    usable_ids = list(set(urate_tmp.eid) - set(withdrawn_ids.ids))
    phenos = urate_tmp[urate_tmp["eid"].isin(usable_ids)]
    del urate_tmp
    # avail_phenos = urate
    geno = geno_tmp[geno_tmp["sample"].isin(usable_ids)]
    del geno_tmp

    print("getting train/test/verify split")
    train_ids, test_ids, verify_ids = get_train_test_verify_ids(phenos, test_frac, verify_frac)

    if (subsample_control):
        print("reducing to even control/non-control split")
        sample_ids = np.concatenate([train_ids, test_ids, verify_ids]).reshape(-1)
        phenos = phenos[phenos["eid"].isin(sample_ids)]
        geno = geno[geno["sample"].isin(sample_ids)]

    train_phenos = phenos[["age", "sex", "bmi"]]

    if summarise_genos:
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
        print("{:.2f}% has gout".format(100 * np.sum(phenos.gout) / len(phenos)))

    if remove_nan:
        geno_mat[np.isnan(geno_mat)] = most_common

    snv_toks = Tokenised_SNVs(geno, encoding)

    return train_phenos, snv_toks, phenos


def get_train_test_verify_ids(phenos, test_frac, verify_frac):
    if exists(test_ids_file) and exists(train_ids_file) and exists(verify_ids_file):
        train_ids = pandas.read_csv(train_ids_file)
        test_ids = pandas.read_csv(test_ids_file)
        verify_ids = pandas.read_csv(verify_ids_file)
    else:
        # usable_ids = phenos.eid.values
        gout = phenos["gout"].values
        test_num_gout = (int)(math.ceil(np.sum(gout) * test_frac))
        verify_num_gout = (int)(math.ceil(np.sum(gout) * verify_frac))
        train_num_gout = np.sum(gout) - test_num_gout - verify_num_gout
        # train_gout_pos = np.random.choice(np.where(gout)[0], train_num_gout, replace=False)
        # non_train_gout_pos = np.setdiff1d(np.where(gout), train_gout_pos)
        # test_gout_pos = np.random.choice(non_train_gout_pos, test_num_gout, replace=False)
        # verify_gout_pos = np.setdiff1d(non_train_gout_pos, test_gout_pos)
        # train_gout_pos.sort()
        # test_gout_pos.sort()
        # verify_gout_pos.sort()

        # normalise w.r.t age, sex & bmi
        num_bins = 50
        # get gout bmi bins
        gout_pos = np.where(gout)[0]
        control_pos = np.where(gout == False)[0]
        gp = phenos.iloc[[i for i in gout_pos]]
        cp = phenos.iloc[[i for i in control_pos]]
        # exclude cases with 'nan' bmi or age
        gp = gp[gp.bmi.isna() == False]
        gp = gp[gp.age.isna() == False]
        cp = cp[cp.bmi.isna() == False]
        cp = cp[cp.age.isna() == False]
        # gout_ages = gp.age.sort_values()
        gp_age_categories = pandas.qcut(gp.age, num_bins, duplicates='drop')
        chosen_control_eids = []
        chosen_gout_eids = []
        total_gout_checked = 0
        for age_cat in gp_age_categories.unique():
            # cp_in_age_cat = cp.iloc[[i in age_cat for i in cp.age]]
            # gp_in_age_cat = gp.iloc[[i in age_cat for i in gp.age]]
            cp_in_age_cat = cp[cp.age.between(age_cat.left, age_cat.right, inclusive='right')]
            gp_in_age_cat = gp[gp.age.between(age_cat.left, age_cat.right, inclusive='right')]
            gp_bmi_categories = pandas.qcut(gp_in_age_cat.bmi, num_bins, duplicates='drop')
            for bmi_cat in gp_bmi_categories.unique():
                # cp_in_age_bmi_cats = cp_in_age_cat.iloc[[i in bmi_cat for i in cp_in_age_cat.bmi]]
                # gp_in_age_bmi_cats = gp_in_age_cat.iloc[[i in bmi_cat for i in gp_in_age_cat.bmi]]
                cp_in_age_bmi_cats = cp_in_age_cat[cp_in_age_cat.bmi.between(bmi_cat.left, bmi_cat.right, inclusive='right')]
                gp_in_age_bmi_cats = gp_in_age_cat[gp_in_age_cat.bmi.between(bmi_cat.left, bmi_cat.right, inclusive='right')]
                for sex in gp.sex.unique():
                    gp_in_age_bmi_sex_cats = gp_in_age_bmi_cats[gp_in_age_bmi_cats.sex == sex]
                    cp_in_age_bmi_sex_cats = cp_in_age_bmi_cats[cp_in_age_bmi_cats.sex == sex]
                    num_gout_this_case = len(gp_in_age_bmi_sex_cats)
                    gout_eids_this_case = gp_in_age_bmi_sex_cats.eid.values
                    available_control_ids = set(cp_in_age_bmi_sex_cats.eid) - set(chosen_control_eids)
                    available_control_ids = [i for i in available_control_ids]
                    control_eids_this_case = np.random.choice(available_control_ids, int(1.0*num_gout_this_case), replace=False)
                    if (len(set(control_eids_this_case).intersection(set(chosen_control_eids))) > 0):
                        print("Warning! sampling ids that are already chosen")
                    if (not len(control_eids_this_case) == num_gout_this_case):
                        print("warning! didnt' sample enough for this case")
                    chosen_control_eids += control_eids_this_case.tolist()
                    chosen_gout_eids += gout_eids_this_case.tolist()
                    total_gout_checked += num_gout_this_case
        new_cp = cp[cp.eid.isin(chosen_control_eids)]
        gp = gp[gp.eid.isin(chosen_gout_eids)]

        print('num gout checked {}, num control {}'.format(total_gout_checked, len(chosen_control_eids)))
        print(gp)
        print(new_cp)


        verify_num_gout = (int)(math.ceil(len(gp) * verify_frac))
        train_num_gout = len(gp) - test_num_gout - verify_num_gout

        gp_left = gp
        train_gout_eid = np.random.choice(gp_left.eid, train_num_gout, replace=False)
        gp_left = gp_left[~gp_left.eid.isin(train_gout_eid)]
        test_gout_eid = np.random.choice(gp_left.eid, test_num_gout, replace=False)
        gp_left = gp_left[~gp_left.eid.isin(test_gout_eid)]
        verify_gout_eid = np.random.choice(gp_left.eid, verify_num_gout, replace=False)

        cp_left = new_cp
        train_control_eid = np.random.choice(cp_left.eid, train_num_gout, replace=False)
        cp_left = cp_left[~cp_left.eid.isin(train_control_eid)]
        test_control_eid = np.random.choice(cp_left.eid, test_num_gout, replace=False)
        cp_left = cp_left[~cp_left.eid.isin(test_control_eid)]
        verify_control_eid = np.random.choice(cp_left.eid, verify_num_gout, replace=False)

        train_gout_eid.sort()
        test_gout_eid.sort()
        verify_gout_eid.sort()
        train_control_eid.sort()
        test_control_eid.sort()
        verify_control_eid.sort()

        if (not len(verify_control_eid) == len(verify_gout_eid)):
            print("Warning! verification sets not balanced: {} (gout) vs. {} (control)".format(len(verify_gout_eid), len(verify_control_eid)))

    #     train_control_pos = np.random.choice(np.where(gout == False)[0], train_num_gout, replace=False)
    #     non_train_control_pos = np.random.choice(np.setdiff1d(np.where(gout == False)[0], train_control_pos), test_num_gout + verify_num_gout, replace=False)
    #     test_control_pos = np.random.choice(non_train_control_pos, test_num_gout, replace=False)
    #     verify_control_pos = np.setdiff1d(non_train_control_pos, test_control_pos)
    #     train_control_pos.sort()
    #     test_control_pos.sort()
    #     verify_control_pos.sort()

        train_all_eids = np.concatenate([train_control_eid, train_gout_eid])
        test_all_eids = np.concatenate([test_control_eid, test_gout_eid])
        verify_all_eids = np.concatenate([verify_control_eid, verify_gout_eid])

        train_ids  = pandas.DataFrame(train_all_eids)
        test_ids   = pandas.DataFrame(test_all_eids)
        verify_ids = pandas.DataFrame(verify_all_eids)
        train_ids.to_csv(train_ids_file, header=["ID"], index=False)
        test_ids.to_csv(test_ids_file, header=["ID"], index=False)
        verify_ids.to_csv(verify_ids_file, header=["ID"], index=False)
    train_ids = np.array(train_ids).reshape(-1)
    test_ids = np.array(test_ids).reshape(-1)
    verify_ids = np.array(verify_ids).reshape(-1)
    return train_ids, test_ids, verify_ids


def get_pretrain_dataset(train_ids, encoding: int = 2):
    print("using data from:", data_dir)
    bed_file = gwas_dir+pretrain_plink_base+".bed"
    bim_file = gwas_dir+pretrain_plink_base+".bim"
    fam_file = gwas_dir+pretrain_plink_base+".fam"
    print("bed_file:", bed_file)
    geno_tmp = read_plink1_bin(bed_file, bim_file, fam_file)
    geno_tmp["sample"] = pandas.to_numeric(geno_tmp["sample"])
    urate_tmp = pandas.read_csv(data_dir + urate_file)
    withdrawn_ids = pandas.read_csv(data_dir + "w12611_20220222.csv", header=None, names=["ids"])

    usable_ids = list(set(train_ids) - set(withdrawn_ids.ids))
    del urate_tmp
    geno = geno_tmp[geno_tmp["sample"].isin(usable_ids)]
    del geno_tmp

    snv_toks = Tokenised_SNVs(geno, encoding)

    return snv_toks

def ids_to_positions(id_list, ids_in_position):
    id_to_pos = {}
    for pos, id in enumerate(ids_in_position):
        id_to_pos[(int)(id)] = pos
    pos_list = [id_to_pos[i] for i in id_list]
    return np.array(pos_list)


def get_train_test_verify(input_phenos, geno, pheno, test_split, verify_split):
    # urate = pheno["urate"].values
    gout = pheno["gout"].values
    # test_cutoff = (int)(math.ceil(test_split * geno.tok_mat.shape[0]))

    train_ids, test_ids, verify_ids = get_train_test_verify_ids(pheno, test_split, verify_split)
    train_all_pos = ids_to_positions(train_ids, geno.ids)
    test_all_pos = ids_to_positions(test_ids, geno.ids)
    verify_all_pos = ids_to_positions(verify_ids, geno.ids)

    positions = geno.positions
    test_seqs = geno.tok_mat[test_all_pos,]
    test_phes = gout[test_all_pos]
    train_seqs = geno.tok_mat[train_all_pos]
    train_phes = gout[train_all_pos]
    verify_seqs = geno.tok_mat[verify_all_pos]
    verify_phes = gout[verify_all_pos]
    print("seqs shape: ", train_seqs.shape)
    print("phes shape: ", train_phes.shape)

    age = input_phenos["age"]
    bmi = input_phenos["bmi"]
    # 0 == Male, 1 == Female
    sex = torch.tensor(input_phenos["sex"].values == "Female", dtype=torch.int64)
    input_pheno_vec = torch.tensor(np.array([
        np.array(age.values),
        np.array(sex),
        np.array(bmi),
    ], dtype=np.int64), dtype=torch.int64).t()
    train_pheno_vec = input_pheno_vec[train_all_pos,]
    test_pheno_vec = input_pheno_vec[test_all_pos,]
    verify_pheno_vec = input_pheno_vec[verify_all_pos,]
    # training_dataset = data.TensorDataset(
    #     tensor(train_seqs, device=device), tensor(train_phes, dtype=torch.int64)
    # )
    # test_dataset = data.TensorDataset(
    #     tensor(test_seqs, device=device), tensor(test_phes, dtype=torch.int64)
    # )
    training_dataset = data.TensorDataset(
        train_pheno_vec, positions.repeat(len(train_seqs), 1), train_seqs, tensor(train_phes, dtype=torch.int64)
    )
    test_dataset = data.TensorDataset(
        test_pheno_vec, positions.repeat(len(test_seqs), 1), test_seqs, tensor(test_phes, dtype=torch.int64)
    )
    verify_dataset = data.TensorDataset(
        verify_pheno_vec, positions.repeat(len(verify_seqs), 1), verify_seqs, tensor(verify_phes, dtype=torch.int64)
    )
    return train_ids, training_dataset, test_ids, test_dataset, verify_ids, verify_dataset

def get_data(enc_ver, test_split, verify_split=0.05):
    train_phenos_file = cache_dir + plink_base + '_encv-' + str(enc_ver) + '_train_phenos_cache.pickle'
    X_file = cache_dir + plink_base + '_encv-' + str(enc_ver) + '_X_cache.pickle'
    Y_file = cache_dir + plink_base +'_encv-' + str(enc_ver) +  '_Y_cache.pickle'
    if exists(X_file) and exists(Y_file) and exists(train_phenos_file):
        with open(X_file, "rb") as f:
            X = pickle.load(f)
        with open(Y_file, "rb") as f:
            Y = pickle.load(f)
        with open(train_phenos_file, "rb") as f:
            train_phenos = pickle.load(f)
    else:
        print("reading data from plink")
        train_phenos, X, Y = read_from_plink(subsample_control=True, encoding=enc_ver)
        print("done, writing to pickle")
        with open(train_phenos_file, "wb") as f:
            pickle.dump(train_phenos, f, pickle.HIGHEST_PROTOCOL)
        with open(X_file, "wb") as f:
            pickle.dump(X, f, pickle.HIGHEST_PROTOCOL)
        with open(Y_file, "wb") as f:
            pickle.dump(Y, f, pickle.HIGHEST_PROTOCOL)
        print("done")

    train_ids, train, test_ids, test, verify_ids, verify = get_train_test_verify(train_phenos, X, Y, test_split, verify_split)
    return train_ids, train, test_ids, test, verify_ids, verify, X, Y, enc_ver

if __name__ == "__main__":
    # snv_toks, urate = read_from_plink(encoding=2)
    # test_split = 0.25
    # verify_split = 0.05
    # train_ids, train, test_ids, test, verify_ids, verify, X, Y, enc_ver = get_data(2, test_split, verify_split)
    # train_ids, test_ids, verify_ids = get_train_test_verify_ids(Y, test_split, verify_split)
    # get_pretrain_dataset(train_ids)

    # debugging get_tok_mat
    enc_ver = 2
    test_frac = 0.25
    verify_frac = 0.05
    geno_tmp = read_plink1_bin("/data/ukbb/all_gwas.bed")
    geno_tmp["sample"] = pandas.to_numeric(geno_tmp["sample"])
    urate_tmp = pandas.read_csv(data_dir + urate_file)
    withdrawn_ids = pandas.read_csv(data_dir + "w12611_20220222.csv", header=None, names=["ids"])

    print("removing withdrawn ids")
    usable_ids = list(set(urate_tmp.eid) - set(withdrawn_ids.ids))
    phenos = urate_tmp[urate_tmp["eid"].isin(usable_ids)]
    del urate_tmp
    # avail_phenos = urate
    geno = geno_tmp[geno_tmp["sample"].isin(usable_ids)]
    del geno_tmp

    print("getting train/test/verify split")
    train_ids, test_ids, verify_ids = get_train_test_verify_ids(phenos, test_frac, verify_frac)

    print("reducing to even control/non-control split")
    sample_ids = np.concatenate([train_ids, test_ids, verify_ids]).reshape(-1)
    phenos = phenos[phenos["eid"].isin(sample_ids)]
    geno = geno[geno["sample"].isin(sample_ids)]

    snv_toks = Tokenised_SNVs(geno, enc_ver)
