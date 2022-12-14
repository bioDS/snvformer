#!/usr/bin/env python3
# from read_data import *
import torch
from torch import nn, tensor
from torch.utils import data
import pickle
from os.path import exists
import numpy as np
import os
from snp_input import get_data, get_pretrain_dataset
# from self_attention_net import *
from snp_network import transformer_from_encoder, get_encoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

import environ

# plink_base = os.environ['PLINK_FILE']
# pretrain_base = os.environ['PRETRAIN_PLINK_FILE']
cache_dir = environ.cache_dir

def get_full_param_string(params: dict):
    str = "\n".join("{}: {}".format(k, v) for k, v in params.items())

def main():
    home_dir = os.environ.get("HOME")
    os.chdir(home_dir + "/work/gout-transformer")
    pretrain_base = environ.pretrain_base

    batch_size = 10
    net_file = "./saved_nets/genotype_p1e-1_encv-2_batch-10_epochs-10_p-65803_n-18776_epoch-10_test_split-0.25_output-tok_net.pickle"

    # Pre-training
    test_frac = 0.25
    verify_frac = 0.05
    output = "tok"
    train_ids, train, test_ids, test, verify_ids, verify, geno, pheno, enc_ver = get_data(2, test_frac, verify_frac)
    pt_pickle = cache_dir + pretrain_base + "_pretrain.pickle"
    if exists(pt_pickle):
        with open(pt_pickle, "rb") as f:
            pretrain_snv_toks = pickle.load(f)
    else:
        pretrain_snv_toks = get_pretrain_dataset(train_ids, enc_ver)
        with open(pt_pickle, "wb") as f:
            pickle.dump(pretrain_snv_toks, f)

    num_phenos = 3
    max_seq_pos = geno.positions.max()
    use_device_ids = [3]
    device = use_device_ids[0]
    encoder = get_encoder(geno.tok_mat.shape[1], num_phenos, max_seq_pos, pretrain_snv_toks.num_toks, batch_size, device, pretrain_snv_toks.string_to_tok['cls'])
    encoder = encoder.cuda(device)
    encoder = nn.DataParallel(encoder, use_device_ids)
    net = transformer_from_encoder(encoder.module, geno.tok_mat.shape[1], num_phenos, output)
    # net = get_transformer(geno.tok_mat.shape[1], num_phenos, max_seq_pos, geno.num_toks, batch_size, device, geno.string_to_tok["cls"], "tok")
    net = nn.DataParallel(net, use_device_ids).to(use_device_ids[0])
    net.load_state_dict(torch.load(net_file))
    net = net.to(device)
    summarise_net(net, test)


def get_contributions(phenos, pos, tX, tY, net, params):
    device = environ.use_device_ids[0]
    num_phenos = params['num_phenos']
    phenos = phenos.unsqueeze(0)
    tX = tX.unsqueeze(0).to(device)
    pos = pos.unsqueeze(0)
    net = net.to(device)
    tYh = net(phenos, tX, pos)
    # for phenos, pos, tX, tY in test_iter:
    #     tX = tX.to(device)
    #     tYh = net(phenos, tX, pos)
    #     break
    # R = net.module.get_relevance_last_class()
    # N.B. resets gradients, don't use during training.
    # (for the moment) only checks the first entry in the batch
    # def get_relevance_last_class(self):
    trainer = torch.optim.SGD(net.module.parameters(), lr=1e-7)
    trainer.zero_grad()
    net.module.last_output[0,1].backward()
    batch_size = net.module.last_output.shape[0]
    # combined_seq_len includes [cls] and phenotypes
    # N.B. R is ~ batch * seq * seq, likely won't fit on device.
    R = torch.eye(net.module.encoder.combined_seq_len)
    # R_init should be (batch_size, seq, seq)
    num_heads = net.module.encoder.num_heads
    # calculate \bar{A} for each layer of the encoder.
    for block in net.module.encoder.blocks:
        A_prod = (block.A.grad[:num_heads,:,:] * block.A[:num_heads,:,:]) # elementwise (\nabla A \cdot A)
        A_prod = A_prod.to('cpu')
        block = block.to('cpu')
        # average positive elements of A_prod
        num_positive = torch.sum(A_prod >= 0.0)
        A_bar = torch.maximum(torch.zeros_like(A_prod), A_prod)
        # shape of A_bar: (batch * heads, seq, lin_k)
        # A_bar = A_bar.reshape(-1, net.module.encoder.num_heads, A_bar.shape[1], A_bar.shape[2])
        # shape of A_bar (batch, num_heads, seq, lin_k)
        A_bar = torch.sum(A_bar, 0) / num_positive
        # shape of A_bar (batch, seq, lin_k)
        # account for linformer (TODO: are we breaking anything here?)
        # we do the same thing to the releveance that we did to the keys.
        linformer_R = block.attention.E_i(R.swapaxes(0,1)).swapaxes(0,1)
        linformer_R = torch.maximum(torch.zeros_like(linformer_R), linformer_R)
        #linformer_R should be (batch, lin_k, seq)
        R[:] = R + torch.matmul(A_bar, linformer_R)
    R_adjusted = R - torch.eye(net.module.encoder.combined_seq_len)
    # N.B. includes [cls] and phenotypes
    cls_contributions = R[0,:]
    cont_sum = torch.sum(cls_contributions[1:])
    pheno_contributions = cls_contributions[1:num_phenos+1] / cont_sum
    seq_contributions = cls_contributions[num_phenos+1:] / cont_sum
    return pheno_contributions, seq_contributions

# doesn''t work for 66k, not emough memory.
# relevance test
# a, b, c, d = test[4500]
# pheno_contributions, seq_contributions = get_contributions(a, b, c, d, net)
# with open("last_seq_contributions.pickle", "wb") as f:
#     pickle.dump((pheno_contributions, seq_contributions), f)
# for ind in range(4501,4550):
#    a, b, c, d = test[ind]
#    ph, se = get_contributions(a, b, c, d, net)
#    pheno_contributions = pheno_contributions + ph
#    seq_contributions = seq_contributions + se
# 
# with open("last_seq_contributions_bundle.pickle", "wb") as f:
#     pickle.dump((pheno_contributions, seq_contributions), f)

# check binary accuracy on test set:


def summarise_net(net, test_data, parameters, net_file):
    # save parameters
    param_string = get_full_param_string(parameters)
    with open(net_file + "_params.txt") as f:
        f.write(param_string)
    batch_size = parameters['batch_size']
    test_iter = data.DataLoader(test_data, (int)(batch_size), shuffle=False)
    actual_vals = []
    predicted_vals = []
    nn_scores = []
    test_loss = 0.0
    loss = nn.CrossEntropyLoss()
    device = environ.use_device_ids[0]
    net = net.cuda(device)
    with torch.no_grad():
        for phenos, pos, tX, tY in test_iter:
            phenos = phenos.to(device)
            pos = pos.to(device)
            tX = tX.to(device)
            tYh = net(phenos, tX, pos)
            tY = tY.to(device)
            l = loss(tYh, tY)
            test_loss += l.mean()
            binary_tYh = tYh[:, 1] > 0.5
            binary_tY = tY > 0.5
            actual_vals.extend(binary_tY.cpu().numpy())
            predicted_vals.extend(binary_tYh.cpu().numpy())
            nn_scores.extend(tYh[:, 1].cpu().numpy())

    print("mean loss: {}", test_loss / len(test_iter))
    test_results = pd.DataFrame({"Score": nn_scores, "Predicted": predicted_vals, "Actual": actual_vals})
    test_results["Correct"] = test_results["Predicted"] == test_results["Actual"]

    print("fraction correct: {:.4f}".format(np.sum(test_results["Correct"]) / len(test_results)))

    pred_gout = test_results[test_results["Predicted"]]
    print("total predicted gout cases: {}".format(len(pred_gout)))
    print("fraction of predicted gout cases correct: {:.4f}".format(np.sum(pred_gout["Correct"]) / len(pred_gout)))

    non_gout = test_results[test_results["Predicted"] == False]
    print("total predicted non gout cases: {}".format(len(non_gout)))
    print("fraction of predicted non-gout cases correct: {:.4f}".format(np.sum(non_gout["Correct"]) / len(non_gout)))

    sns.set_theme(style="whitegrid")
    # distribution of scores
    plt.close()
    fig, ax = plt.subplots(figsize=(3, 2))
    fig = sns.histplot(
        data=test_results, x="Score", bins=30, kde=True
    )
    # ax.set_xticks([0.0, 0.3, 0.5, 0.8])
    ax.set_xticks([0.1 * i for i in range(11)])
    ax.set_xticklabels([i for i in range(11)])
    ax.set_xlabel("Score $\\times 10^{-1}$")

    plt.tight_layout()
    plt.savefig(net_file + "score_dist.pdf")
    plt.savefig(net_file + "score_dist.png")

    # score vs. accuracy
    # min_score = np.min(nn_scores)
    # max_score = np.max(nn_scores)
    # bins = np.arange(min_score, max_score, (max_score-min_score)/10)
    # test_results.sort_values
    # test_results["Score (range)"], bins = pd.qcut(test_results["Score"], 10, precision=1, retbins=True)
    test_results["Score (range)"], bins = pd.cut(test_results["Score"], [0.1 * i for i in range(11)], precision=1, retbins=True)
    # bin_titles = ["{:.0f}".format(10*i) for i in bins]
    bin_titles = ["{:.0f}-{:.0f}".format(10 * i, 10 * i + 1) for i in bins]
    test_results["Correct (%)"] = 100 * test_results["Correct"]

    plt.close()
    fig, ax = plt.subplots(figsize=(4.5, 2))
    fig = sns.pointplot(
        data=test_results, x="Score (range)", y="Correct (%)", ci=95, n_boot=1000,
        markers='.'
    )
    for i, line in enumerate(ax.get_lines()):
        if (i > 0):
            line.set_alpha(0.5)
    # fig = sns.barplot(
    #     data=test_results, x="Score (range)", y="Correct", color="b"
    # )
    # ax.set_ylim((0.4, 0.8))
    ax.set_xticklabels(bin_titles[:-1])
    ax.set_xlabel("Score $\\times 10^{-1}$")
    # ax.set_yticks([0.5, 0.7])

    plt.tight_layout()
    plt.savefig(net_file + "score_vs_tp.pdf")
    plt.savefig(net_file + "score_vs_tp.png")

    # AUROC
    auroc = metrics.roc_auc_score(actual_vals, nn_scores)
    fpr, tpr, _ = metrics.roc_curve(actual_vals, nn_scores)
    print("auroc: {}".format(auroc))

    plt.close()
    plt.figure()
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        label="ROC curve (area = %0.2f)" % auroc,
    )
    plt.plot([0, 1], [0, 1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend(loc="lower right")
    plt.show()
    plt.tight_layout()
    plt.savefig(net_file + "roc.pdf")
    plt.savefig(net_file + "roc.png")
    print("saved in {}".format(net_file))


if __name__ == "__main__":
    main()
