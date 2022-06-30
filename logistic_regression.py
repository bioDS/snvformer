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


class secretly_logistic(nn.Module):
    def __init__(self, num_phenos):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(num_phenos, 2),
            nn.Softmax(dim=-1)
        )
        self.linear.apply(init_weights)

    def forward(self, phenos, x, pos):
        lin_out = self.linear(phenos.float())
        return lin_out

def get_logistic_model(num_phenos):
    net = secretly_logistic(num_phenos)
    return net


def train_logistic_mlp(params, net, train_set, test_set):
    loss = nn.CrossEntropyLoss()
    all_phenos, _, _, gout = train_set[:]
    test_phenos, _, _, test_gout = test_set[:]
    test_phenos = test_phenos.cuda()
    test_gout = test_gout.cuda()
    all_phenos = all_phenos.cuda()
    gout = gout.cuda()
    net = net.cuda()
    train_subset = torch.utils.data.TensorDataset(all_phenos, gout)
    test_subset = torch.utils.data.TensorDataset(test_phenos, test_gout)
    train_subset = torch.utils.data.TensorDataset(all_phenos, gout)
    training_iter = torch.utils.data.DataLoader(train_subset, params['batch_size'], shuffle=True)
    # test_iter = torch.utils.data.DataLoader(test_subset, params['batch_size'], shuffle=True)
    trainer = torch.optim.SGD(net.parameters(), lr=params['lr'])
    # trainer = torch.optim.AdamW(net.parameters(), lr=params['lr'], amsgrad=True)
    for e in tqdm(range(params['num_epochs']), ncols=0):
        # for phenos, pos, X, y in tqdm(training_iter, ncols=0):
        # for phenos, pos, X, y in training_iter:
        for phenos, y in training_iter:
            # phenos = phenos.cuda()
            # y = y.cuda()
            pred = net(phenos, None, None) 
            l = loss(pred, y)
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()

    print("random test cases:")
    choice = np.random.choice(len(test_subset), 10)
    phenos, y = test_subset[choice]
    with torch.no_grad():
        pred = net(phenos, None, None) 
    for s, a, b in zip(phenos, pred, y):
        print("(s: {}) - {:.2f}: {}".format(s[1], a[1], b))

# logistic regression of age & sex to determine gout:
def logistic():
    params = snp_network.default_parameters
    params['batch_size'] = 256
    params['lr'] = 1e-5
    params['encoding_version'] = 6
    params['num_epochs'] = 500
    train_ids, train, test_ids, test, verify_ids, verify, geno, pheno, enc_ver = get_data(params)
    params['num_phenos'] = num_phenos = 3
    net = get_logistic_model(num_phenos) # all three for now
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


if __name__ == "__main__":
    logistic()
    mlp()
