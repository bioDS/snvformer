#!/usr/bin/env python3
import snp_network
import os
import torch
from snp_input import get_data, get_pretrain_dataset
from torch import nn
from net_test_summary import summarise_net
from snp_network import get_net_savename
from environ import saved_nets_dir


def geno_only():
    parameters = snp_network.default_parameters
    parameters['pretrain_base'] = 'all_unimputed_combined'
    parameters['plink_base'] = 'genotyped_p1e-1'
    parameters['continue_training'] = False
    parameters['train_new_encoder'] = True
    parameters['use_device_ids'] = [4]
    parameters['encoding_version'] = 2
    parameters['test_frac'] = 0.25
    parameters['verify_frac'] = 0.05
    parameters['batch_size'] = 5
    parameters['num_epochs'] = 20
    parameters['lr'] = 1e-7
    parameters['pt_lr'] = 1e-7
    parameters['encoder_size'] = 65803  # the size of gneotyped_p1e-1
    parameters['pretrain_epochs'] = 0  # original version has no pre-training
    parameters['num_phenos'] = 0  # exclude phenotypes
    parameters['use_phenos'] = False
    parameters['output_type'] = 'tok'
    parameters['embed_dim'] = 64
    parameters['num_heads'] = 4
    parameters['num_layers'] = 4
    parameters['linformer_k'] = 64
    parameters['use_linformer'] = True

    # TODO loading all this here is overkill
    train_ids, train, test_ids, test, verify_ids, verify, geno, pheno, enc_ver = get_data(parameters)
    pretrain_snv_toks = get_pretrain_dataset(train_ids, parameters)

    # reload trained network if it exists
    net_file = saved_nets_dir + get_net_savename(parameters)
    if os.path.exists(net_file):
        print("reloading file: {}".format(net_file))
        net = snp_network.get_transformer_from_params(parameters, pretrain_snv_toks)
        net = nn.DataParallel(net, parameters['use_device_ids'])
        net.load_state_dict(torch.load(net_file))
        net = net.to(parameters['use_device_ids'][0])
    else:
        print("Training new net, no saved net in file '{}'".format(net_file))
        net = snp_network.train_everything(parameters)

    print("summarising test-set results")
    summarise_net(net, test, parameters, net_file)


def pheno_v1():
    parameters = snp_network.default_parameters
    parameters['pretrain_base'] = 'all_unimputed_combined'
    parameters['plink_base'] = 'genotyped_p1e-1'
    parameters['continue_training'] = False
    parameters['train_new_encoder'] = True
    parameters['use_device_ids'] = [4]
    parameters['encoding_version'] = 2
    parameters['test_frac'] = 0.25
    parameters['verify_frac'] = 0.05
    parameters['batch_size'] = 5
    parameters['num_epochs'] = 20
    parameters['lr'] = 1e-7
    parameters['pt_lr'] = 1e-7
    parameters['encoder_size'] = 65803  # the size of gneotyped_p1e-1
    parameters['pretrain_epochs'] = 0  # original version has no pre-training
    parameters['num_phenos'] = 3
    parameters['use_phenos'] = True
    parameters['output_type'] = 'tok'
    parameters['embed_dim'] = 96
    parameters['num_heads'] = 4
    parameters['num_layers'] = 6
    parameters['linformer_k'] = 96
    parameters['use_linformer'] = True

    # TODO loading all this here is overkill
    train_ids, train, test_ids, test, verify_ids, verify, geno, pheno, enc_ver = get_data(parameters)
    pretrain_snv_toks = get_pretrain_dataset(train_ids, parameters)
    #parameters['max_seq_pos'] = pretrain_snv_toks.positions.max()
    #parameters['output_type'] = 'tok'
    #parameters['num_toks'] = pretrain_snv_toks.num_toks

    # reload trained network if it exists
    net_file = saved_nets_dir + get_net_savename(parameters)
    if os.path.exists(net_file):
        print("reloading file: {}".format(net_file))
        net = snp_network.get_transformer(parameters, pretrain_snv_toks)
        net = nn.DataParallel(net, parameters['use_device_ids'])
        net.load_state_dict(torch.load(net_file))
        net = net.to(parameters['use_device_ids'][0])
    else:
        print("Training new net, no saved net in file '{}'".format(net_file))
        net = snp_network.train_everything(parameters)

    print("summarising test-set results")
    summarise_net(net, test, parameters, net_file)


if __name__ == "__main__":
    home_dir = os.environ.get("HOME")
    os.chdir(home_dir + "/work/gout-transformer")

    geno_only()
    # pheno_v1() # done
