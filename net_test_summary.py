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
from snp_network import get_data, get_transformer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plink_base = os.environ['PLINK_FILE']

home_dir = os.environ.get("HOME")
os.chdir(home_dir + "/work/gout-transformer")

batch_size = 30
num_epochs = 50
train, test, geno, pheno, enc_ver = get_data()
# net_name = "{}_ai-{}_batch-{}_epochs-{}_p-{}_n-{}_controlx-{}_net.pickle".format(
#     plink_base, str(use_ai_encoding), batch_size, num_epochs, geno.tok_mat.shape[1], geno.tok_mat.shape[0], control_scale
# )
net_file = "ai-True_batch-180_epochs-150_p-65803_n-18776_net.pickle"

max_seq_pos = geno.positions.max()
device = torch.device('cuda:7')
use_device_ids = [7]
net = get_transformer(geno.tok_mat.shape[1], max_seq_pos, geno.num_toks, batch_size, device)
net = nn.DataParallel(net, use_device_ids).to(use_device_ids[0])
net.load_state_dict(torch.load(net_file))
net.to(device)

# check binary accuracy on test set:
# test_gout_correct = 0
# test_gout_incorrect = 0
# test_control_correct = 0
# test_control_incorrect = 0
test_iter = data.DataLoader(test, (int)(batch_size), shuffle=False)
actual_vals = []
predicted_vals = []
nn_scores = []
with torch.no_grad():
    for pos, tX, tY in test_iter:
        tX = tX.to(device)
        tYh = net(tX, pos)
        binary_tYh = tYh[:, 1] > 0.5
        binary_tY = tY > 0.5
        actual_vals.extend(binary_tY.cpu().numpy())
        predicted_vals.extend(binary_tYh.cpu().numpy())
        nn_scores.extend(tYh[:, 1].cpu().numpy())

test_results = pd.DataFrame({"Score": nn_scores, "Predicted": predicted_vals, "Actual": actual_vals})
test_results["Correct"] = test_results["Predicted"] == test_results["Actual"]

# distribution of scores
plt.close()
fig = sns.histplot(
    data=test_results, x="Score", bins=30
)

plt.savefig("score_dist.svg")
plt.savefig("score_dist.png")

# score vs. accuracy
# min_score = np.min(nn_scores)
# max_score = np.max(nn_scores)
# bins = np.arange(min_score, max_score, (max_score-min_score)/10)
# test_results.sort_values
test_results["Score (range)"] = pd.qcut(test_results["Score"], 10)

plt.close()
fig = sns.pointplot(
    data=test_results, x="Score (range)", y="Correct"
)

plt.savefig("score_vs_tp.svg")
plt.savefig("score_vs_tp.png")
