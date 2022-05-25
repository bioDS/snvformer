#!/usr/bin/env python3
# from read_data import *
import torch
from torch import nn, tensor
from torch.utils import data
import math
import pickle
from os.path import exists
import numpy as np
import os
from snp_input import *
from self_attention_net import *
from snp_network import get_data, get_transformer, transformer_from_encoder, get_encoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

plink_base = os.environ['PLINK_FILE']
pretrain_base = os.environ['PRETRAIN_PLINK_FILE']

home_dir = os.environ.get("HOME")
os.chdir(home_dir + "/work/gout-transformer")

batch_size = 10
# num_epochs = 50
# train, test, geno, pheno, enc_ver = get_data()
# net_file = "genotyped_p1e-1_encv-3_batch-180_epochs-1_p-65803_n-18776_net.pickle"
# net_file = "genotyped_p1e-1_encv-2_batch-180_epochs-100_p-65803_n-18776_net.pickle"
# net_file = "genotyped_p1e-1_encv-3_batch-180_epochs-100_p-65803_n-18776_net.pickle"
# net_file = "genotyped_p1e-1_encv-2_batch-180_epochs-150_p-65803_n-18776_net.pickle"
# net_file = "genotyped_p1e-1_encv-2_batch-90_epochs-300_p-65803_n-18776_epoch-300_net.pickle"
# net_file = "66k_gwas_encv-2_batch-180_epochs-150_p-65803_n-18776_net.pickle"
# net_file = "use_net.pickle"
# net_file = "66k_gwas_encv-2_batch-20_epochs-100_p-65803_n-18776_epoch-100_test_split-0.05_net.pickle"
# net_file = "66k_gwas_encv-2_batch-90_epochs-150_p-65803_n-18776_epoch-300_net.pickle"
# net_file = "net_epochs/66k_gwas_batch-60_epoch-240_test_split-0.3_net.pickle"
# net_file = "./66k_gwas_encv-2_batch-60_epochs-200_p-65803_n-18776_epoch-400_output-binary_net.pickle"
net_file = "./net_epochs/genotyped_p1e-1_batch-10_epoch-80_test_split-0.25_net.pickle"
# net_file = "./saved_nets/genotyped_p1e-1_encv-2_batch-10_epochs-100_p-65803_n-18776_epoch-100_test_split-0.25_output-tok_net.pickle"
# net_file = "66k_gwas_batch-90_epoch-210_net.pickle"
# net_file = "66k_gwas_batch-90_epoch-210_net.pickle"
test_file = "./saved_nets/genotyped_p1e-1_encv-2_batch-10_epochs-100_p-65803_n-18776_epoch-100_test_split-0.25_output-tok_net.pickle_test.pickle"
# test_file = net_file = "_test.pickle"
#with open(test_file, "rb") as f:
#    test = pickle.load(f)
#with open("cache/genotyped_p1e-1_encv-2_X_cache.pickle", "rb") as f:
## with open("cache/66k_gwas_encv-2_geno_cache.pickle", "rb") as f:
#    geno = pickle.load(f)

# net_name = "{}_ai-{}_batch-{}_epochs-{}_p-{}_n-{}_controlx-{}_net.pickle".format(
#     plink_base, str(use_ai_encoding), batch_size, num_epochs, geno.tok_mat.shape[1], geno.tok_mat.shape[0], control_scale
# )
# net_file = "ai-True_batch-180_epochs-150_p-65803_n-18776_net.pickle"
# net_file = "new_enc_50_epochs_0596_test_network.pickle"

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
use_device_ids = [5]
device = use_device_ids[0]
encoder = get_encoder(geno.tok_mat.shape[1], num_phenos, max_seq_pos, pretrain_snv_toks.num_toks, batch_size, device, pretrain_snv_toks.string_to_tok['cls'])
encoder = encoder.cuda(device)
encoder = nn.DataParallel(encoder, use_device_ids)
net = transformer_from_encoder(encoder.module, geno.tok_mat.shape[1], num_phenos, output)
# net = get_transformer(geno.tok_mat.shape[1], num_phenos, max_seq_pos, geno.num_toks, batch_size, device, geno.string_to_tok["cls"], "tok")
net = nn.DataParallel(net, use_device_ids).to(use_device_ids[0])
net.load_state_dict(torch.load(net_file))
net = net.to(device)

# check binary accuracy on test set:
test_iter = data.DataLoader(test, (int)(batch_size), shuffle=False)
actual_vals = []
predicted_vals = []
nn_scores = []
test_loss = 0.0
loss = nn.CrossEntropyLoss()
with torch.no_grad():
    for phenos, pos, tX, tY in test_iter:
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

print("mean loss: {}", test_loss/len(test_iter))
test_results = pd.DataFrame({"Score": nn_scores, "Predicted": predicted_vals, "Actual": actual_vals})
test_results["Correct"] = test_results["Predicted"] == test_results["Actual"]

print("fraction correct: {:.4f}".format(np.sum(test_results["Correct"])/len(test_results)))

pred_gout = test_results[test_results["Predicted"]]
print("total predicted gout cases: {}".format(len(pred_gout)))
print("fraction of predicted gout cases correct: {:.4f}".format(np.sum(pred_gout["Correct"])/len(pred_gout)))

non_gout = test_results[test_results["Predicted"] == False]
print("total predicted non gout cases: {}".format(len(non_gout)))
print("fraction of predicted non-gout cases correct: {:.4f}".format(np.sum(non_gout["Correct"])/len(non_gout)))

sns.set_theme(style="whitegrid")
# distribution of scores
plt.close()
fig, ax = plt.subplots(figsize=(3,2))
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
bin_titles = ["{:.0f}-{:.0f}".format(10*i, 10*i + 1) for i in bins]
test_results["Correct (%)"] = 100*test_results["Correct"]

plt.close()
fig, ax = plt.subplots(figsize=(4.5,2))
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
fig, ax = plt.subplots(figsize=(4,4))
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