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

# TODO: should we do this elsewhere?
def init_weights(m):
   if type(m) == nn.Linear:
       nn.init.normal_(m.weight, std=0.01)

class secretly_logistic(nn.Module):
    def __init__(self, num_phenos):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(num_phenos, 2))
        # self.linear.weight.data.normal_(0, 0.01)
        self.sm = nn.Softmax()
        self.linear.apply(init_weights)

    def forward(self, phenos, x, pos):
        lin_out = self.linear(phenos.float())
        return self.sm(lin_out)
        return lin_out

def get_logistic_model(num_phenos):
    net = secretly_logistic(num_phenos)
    return net


def train_logistic(params, net, train_set):
    loss = nn.CrossEntropyLoss()
    training_iter = torch.utils.data.DataLoader(train_set, params['batch_size'], shuffle=True)
    trainer = torch.optim.SGD(net.parameters(), lr=params['lr'])
    for e in tqdm(range(params['num_epochs']), ncols=0):
        # for phenos, pos, X, y in tqdm(training_iter, ncols=0):
        for phenos, pos, X, y in training_iter:
            pred = net(phenos, X, pos)
            l = loss(pred, y)
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()

    for phenos, pos, X, y in tqdm(training_iter, ncols=0):
        break

# logistic regression of age & sex to determine gout:
def logistic():
    params = snp_network.default_parameters
    params['batch_size'] = 512
    params['encoding_version'] = 6
    train_ids, train, test_ids, test, verify_ids, verify, geno, pheno, enc_ver = get_data(params)
    params['num_phenos'] = num_phenos = 3
    net = get_logistic_model(num_phenos) # all three for now
    train_logistic(params, net, train)
    net_file = environ.saved_nets_dir + "logistic_{}val".format(num_phenos)

    summarise_net(net, test, params, net_file)


# simple mlp of age & sex to determine gout:


if __name__ == "__main__":
    logistic()
