#!/usr/bin/env python3
import snp_network
import os
import torch
import environ
from snp_input import get_data, get_pretrain_dataset
from torch import nn
from net_test_summary import summarise_net
from snp_network import get_net_savename
from environ import saved_nets_dir
from tqdm import tqdm
import numpy as np
from torch.utils import data
import seaborn as sns
import pandas
import matplotlib.pyplot as plt

# TODO: should we do this elsewhere?
def init_weights(m):
   if type(m) == nn.Linear:
       nn.init.normal_(m.weight, std=0.01)

class secretly_mlp(nn.Module):
    def __init__(self, num_phenos, num_hiddens):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(num_phenos, num_hiddens),
            nn.GELU(),
            nn.Linear(num_hiddens, 2),
            nn.Softmax(dim=-1)
        )
        self.linear.apply(init_weights)

    def forward(self, phenos, x, pos):
        lin_out = self.linear(phenos.float())
        return lin_out

def get_mlp_model(num_phenos, num_hidden):
    net = secretly_mlp(num_phenos, num_hidden)
    return net

class tin_geno_pheno(nn.Module):
    def __init__(self, num_geno, num_pheno):
        super().__init__()
        self.geno_linear = nn.Sequential(
            nn.Linear(num_geno, 1, bias=False),
        )
        self.pheno_linear = nn.Sequential(
            nn.Linear(num_pheno, 1, bias=True),
        )

    def forward(self, phenos, x, pos):
        lin_out = self.geno_linear(x.float()) + self.pheno_linear(phenos.float())
        lin_out = lin_out.repeat([1, 2])
        lin_out[:,0] = 1-lin_out[:,1]
        return lin_out

class logistic_geno(nn.Module):
    def __init__(self, num_phenos, init=True):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(num_phenos, 1, bias=False),
            # nn.Sigmoid(),
        )
        if (init):
            self.linear.apply(init_weights)

    def forward(self, phenos, x, pos):
        lin_out = self.linear(x.float())
        lin_out = lin_out.unsqueeze(1)
        lin_out = lin_out.repeat([1, 2])
        lin_out[:,0] = 1-lin_out[:,1]
        # print(lin_out.shape)
        # quit()
        return lin_out


class secretly_logistic_geno_only(nn.Module):
    def __init__(self, num_phenos):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(num_phenos, 1),
            nn.Sigmoid(),
        )
        self.linear.apply(init_weights)

    def forward(self, phenos, x, pos):
        lin_out = self.linear(x.float())
        lin_out = lin_out.repeat([1, 2])
        lin_out[:,0] = 1-lin_out[:,1]
        return lin_out

class secretly_logistic(nn.Module):
    def __init__(self, num_phenos):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(num_phenos, 1),
            nn.Sigmoid(),
        )
        self.linear.apply(init_weights)

    def forward(self, phenos, x, pos):
        lin_out = self.linear(phenos.float())
        lin_out = lin_out.repeat([1, 2])
        lin_out[:,0] = 1-lin_out[:,1]
        return lin_out

def get_logistic_model(num_phenos):
    net = secretly_logistic(num_phenos)
    return net

tin_weights = np.array([
    0.024,
    0.024,
    0.100,
    0.133,
    0.058,
    -0.036,
    0.053,
    -0.043,
    -0.025,
    0.023,
    0.035,
    0.025,
    -0.056,
    0.070,
    0.052,
    0.022,
    0.041,
    -0.022,
    0.032,
    -0.048,
    -0.022,
    0.038,
    -0.086,
    -0.024,
    -0.023,
    0.045,
    0.048,
    -0.021,
    -0.028,
    -0.028,
    -0.062,
    -0.022,
    0.215,
    0.330,
    0.107,
    0.022,
    0.024,
    0.093,
    0.254,
    0.023,
    -0.023,
    -0.028,
    -0.027,
    -0.042,
    -0.074,
    0.066,
    0.062,
    -0.091,
    -0.067,
    -0.057,
    0.053,
    -0.059,
    0.051,
    -0.048,
    0.039,
    0.029,
    0.046,
    0.030,
    0.042,
    0.049,
    0.023,
    0.030,
    0.034,
    -0.022,
    0.041,
    0.045,
    -0.024,
    0.031,
    0.061,
    -0.037,
    0.064,
    -0.039,
    0.079,
    -0.038,
    0.025,
    -0.029,
    0.030,
    0.079,
    -0.048,
    -0.033,
    0.030,
    0.025,
    0.029,
    -0.076,
    0.046,
    0.032,
    -0.028,
    -0.081,
    0.032,
    0.042,
    -0.024,
    0.026,
    0.024,
    -0.026,
    -0.029,
    0.025,
    -0.028,
    -0.054,
    -0.040,
    0.023,
    0.046,
    -0.030,
    0.025,
    0.041,
    -0.026,
    -0.028,
    -0.036,
    0.022,
    0.025,
    0.038,
    0.050,
    -0.030,
    0.029,
    0.039,
    -0.024,
    -0.027,
    0.025,
    -0.118,
    0.023,
    -0.023,
    -0.076,
    -0.025,
    -0.033
])

def get_tin_pheno_weights_model():
    geno_weights = torch.tensor(tin_weights, dtype=torch.float32)
    net = tin_geno_pheno(len(geno_weights), 3)
    net.geno_linear[0].weight.data = geno_weights
    # net.pheno_linear[0].weight.data = pheno_weights
    net.geno_linear[0].requires_grad_(False)
    return net


def get_tin_weights_model():
    weights = torch.tensor(tin_weights, dtype=torch.float32)
    print(len(weights))
    net = logistic_geno(len(weights), init=False)
    net.linear[0].weight.data = weights
    return net

def train_cpu(params, net, train_set, test_set):
    loss = nn.CrossEntropyLoss()
    training_iter = torch.utils.data.DataLoader(train_set, params['batch_size'], shuffle=True)
    trainer = torch.optim.SGD(net.parameters(), lr=params['lr'])
    for e in tqdm(range(params['num_epochs']), ncols=0):
        for phenos, pos, X, y in training_iter:
            pred = net(phenos, X, pos) 
            l = loss(pred, y)
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()

def train_logistic_mlp(params, net, train_set, test_set):
    loss = nn.CrossEntropyLoss()
    all_phenos, _, _, gout = train_set[:]
    test_phenos, _, _, test_gout = test_set[:]
    test_phenos = test_phenos.cuda()
    test_gout = test_gout.cuda()
    all_phenos = all_phenos.cuda()
    gout = gout.cuda()
    net = net.cuda()
    # train_subset = torch.utils.data.TensorDataset(all_phenos, gout)
    # test_subset = torch.utils.data.TensorDataset(test_phenos, test_gout)
    # train_subset = torch.utils.data.TensorDataset(all_phenos, gout)
    training_iter = torch.utils.data.DataLoader(train_set, params['batch_size'], shuffle=True)
    # training_iter = torch.utils.data.DataLoader(train_subset, params['batch_size'], shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_set, params['batch_size'], shuffle=True)
    # trainer = torch.optim.SGD(net.parameters(), lr=params['lr'])
    trainer = torch.optim.AdamW(net.parameters(), lr=params['lr'], amsgrad=True)
    for e in tqdm(range(params['num_epochs']), ncols=0):
        # for phenos, pos, X, y in tqdm(training_iter, ncols=0):
        sum_loss = 0.0
        for phenos, pos, X, y in training_iter:
        # for phenos, y in training_iter:
            X = X.cuda()
            pos = pos.cuda()
            phenos = phenos.cuda()
            y = y.cuda()
            pred = net(phenos, X, pos) 
            l = loss(pred, y)
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()
            sum_loss += l
        if e % (params['num_epochs'] / 10) == 0:
            test_loss = 0.0
            with torch.no_grad():
                for phenos, pos, X, y in test_iter:
                    X = X.cuda()
                    y = y.cuda()
                    pos = pos.cuda()
                    phenos = phenos.cuda()
                    Yh = net(phenos, X, pos)  # two-value softmax (binary classification)
                    l = loss(Yh, y)
                    test_loss += l.mean()
            tmpstr = "epoch {}, mean loss {:.5}, {:.5} (test)".format(
                    e, sum_loss / len(training_iter), test_loss / len(test_iter))
            print(tmpstr)

    # print("random test cases:")
    # choice = np.random.choice(len(test_subset), 10)
    # phenos, y = test_subset[choice]
    # with torch.no_grad():
    #     pred = net(phenos, None, None) 
    # for s, a, b in zip(phenos, pred, y):
    #     print("(s: {}) - {:.2f}: {}".format(s[1], a[1], b))

def remove_bmi(dataset):
    phenos, pos, x, y = dataset[:]
    new_phenos = phenos[:,0:2]
    new_dataset = torch.utils.data.TensorDataset(new_phenos, pos, x, y)
    return new_dataset

# logistic regression of age & sex to determine gout:
def logistic():
    params = snp_network.default_parameters
    params['batch_size'] = 256
    params['lr'] = 1e-2
    params['encoding_version'] = 5
    params['num_epochs'] = 100
    params['pretrain_base'] = 'tin_fixed_order'
    params['plink_base'] = 'tin_fixed_order'
    train_ids, train, test_ids, test, verify_ids, verify, geno, pheno, enc_ver = get_data(params)
    params['num_phenos'] = num_phenos = 2
    train = remove_bmi(train)
    test = remove_bmi(test)
    net = get_logistic_model(num_phenos) # all three for now
    train_logistic_mlp(params, net, train, test)
    net_file = environ.saved_nets_dir + "logistic_{}val".format(num_phenos)
    summarise_net(net, test, params, net_file)


def logistic_geno_only():
    params = snp_network.default_parameters
    params['batch_size'] = 16
    params['lr'] = 1e-5
    params['encoding_version'] = 6
    params['num_epochs'] = 50
    train_ids, train, test_ids, test, verify_ids, verify, geno, pheno, enc_ver = get_data(params)
    params['num_phenos'] = num_phenos = 3
    net = secretly_logistic_geno_only(geno.tok_mat.shape[1])
    # net = nn.DataParallel(net)
    train_logistic_mlp(params, net, train, test)
    net_file = environ.saved_nets_dir + "logistic_{}val".format(num_phenos)
    summarise_net(net, test, params, net_file)


# simple mlp of age & sex to determine gout:
def mlp():
    params = snp_network.default_parameters
    params['batch_size'] = 10
    params['lr'] = 0.1
    params['encoding_version'] = 6
    params['num_epochs'] = 10
    train_ids, train, test_ids, test, verify_ids, verify, geno, pheno, enc_ver = get_data(params)
    params['num_phenos'] = num_phenos = 3
    params['num_hiddens'] = num_hiddens = 8
    net = get_mlp_model(num_phenos, num_hiddens)
    train_logistic_mlp(params, net, train, test)
    net_file = environ.saved_nets_dir + "mlp_{}val".format(num_phenos)
    summarise_net(net, test, params, net_file)

def add_prs_score_to_dataset(dataset, prs_net):
    phenos, pos, x, y = dataset[:]
    prs_scores = torch.unsqueeze(prs_net(phenos, x, pos)[:,1], 1)
    new_phenos = torch.cat([phenos, prs_scores], 1)
    new_dataset = torch.utils.data.TensorDataset(new_phenos, pos, x, y)
    return new_dataset


def prs_score_logistic():
    params = snp_network.default_parameters
    params['batch_size'] = 64
    params['lr'] = 1e-2
    params['encoding_version'] = 5
    params['num_epochs'] = 100
    params['pretrain_base'] = 'tin_fixed_order'
    params['plink_base'] = 'tin_fixed_order'
    train_ids, train, test_ids, test, verify_ids, verify, geno, pheno, enc_ver = get_data(params)
    params['num_phenos'] = num_phenos = 3

    # # add tin prs to train and test sets.
    with torch.no_grad():
        prs_net = get_tin_weights_model()
        train = remove_bmi(train)
        test = remove_bmi(test)
        train = add_prs_score_to_dataset(train, prs_net)
        test = add_prs_score_to_dataset(test, prs_net)

    net = secretly_logistic(num_phenos)
    train_logistic_mlp(params, net, train, test)
    # train_cpu(params, net, train, test)
    net_file = environ.saved_nets_dir + "pheno_tin_scores"
    summarise_net(net, test, params, net_file)


def tin_pheno_prs():
    params = snp_network.default_parameters
    params['batch_size'] = 256
    params['lr'] = 1e-5
    params['encoding_version'] = 5
    params['num_epochs'] = 100
    params['pretrain_base'] = 'tin_fixed_order'
    params['plink_base'] = 'tin_fixed_order'
    params['num_phenos'] = 3
    train_ids, train, test_ids, test, verify_ids, verify, geno, pheno, enc_ver = get_data(params)
    net = get_tin_pheno_weights_model()
    train_logistic_mlp(params, net, train, test)
    net_file = environ.saved_nets_dir + "pheno_tin"
    summarise_net(net, test, params, net_file)

def tin_prs():
    params = snp_network.default_parameters
    params['batch_size'] = 256
    params['lr'] = 1e-5
    params['encoding_version'] = 5
    params['num_epochs'] = 0
    params['pretrain_base'] = 'tin_fixed_order'
    params['plink_base'] = 'tin_fixed_order'
    params['num_phenos'] = 3
    train_ids, train, test_ids, test, verify_ids, verify, geno, pheno, enc_ver = get_data(params)
    net = get_tin_weights_model()
    net_file = environ.saved_nets_dir + "reuse_tin"
    summarise_net(net, test, params, net_file)

if __name__ == "__main__":
    # logistic()
    # mlp()
    # tin_prs()
    # tin_pheno_prs()
    # logistic_geno_only()
    prs_score_logistic()
