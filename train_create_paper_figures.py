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


def get_pheno_only_params():
    parameters = snp_network.default_parameters
    parameters['pretrain_base'] = 'all_unimputed_combined'
    parameters['plink_base'] = 'genotyped_p1e-1'
    parameters['continue_training'] = False
    parameters['train_new_encoder'] = False
    parameters['encoding_version'] = 6
    parameters['test_frac'] = 0.25
    parameters['verify_frac'] = 0.05
    parameters['batch_size'] = 5
    parameters['num_epochs'] = 50
    parameters['lr'] = 1e-7
    parameters['pt_lr'] = 1e-7
    parameters['encoder_size'] = 65803  # the size of gneotyped_p1e-1
    parameters['pretrain_epochs'] = 0
    parameters['num_phenos'] = 3
    parameters['use_phenos'] = True
    parameters['output_type'] = 'tok'
    parameters['embed_dim'] = 64
    parameters['num_heads'] = 4
    parameters['num_layers'] = 4
    parameters['linformer_k'] = 64
    parameters['use_linformer'] = True
    parameters['input_filtering'] = 'random_test_verify'
    parameters['mask_genotypes'] = True
    return parameters

def pheno_only():
    parameters = get_pheno_only_params()
    train_ids, train, test_ids, test, verify_ids, verify, geno, pheno, enc_ver = get_data(parameters)
    net_file = saved_nets_dir + get_net_savename(parameters)
    net, pretrain_snv_toks = get_or_train_net(parameters, train_ids)
    print("summarising test-set results")
    summarise_net(net, test, parameters, net_file)

def get_geno_only_params():
    parameters = snp_network.default_parameters
    parameters['pretrain_base'] = 'all_unimputed_combined'
    parameters['plink_base'] = 'genotyped_p1e-1'
    parameters['continue_training'] = False
    parameters['train_new_encoder'] = True
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
    parameters['input_filtering'] = 'random_test_verify'
    return parameters


def get_v6_params():
    parameters = snp_network.default_parameters
    parameters['pretrain_base'] = 'all_unimputed_combined'
    parameters['plink_base'] = 'genotyped_p1e-1'
    parameters['continue_training'] = False
    parameters['train_new_encoder'] = False
    parameters['encoding_version'] = 6
    parameters['test_frac'] = 0.25
    parameters['verify_frac'] = 0.05
    parameters['batch_size'] = 5
    parameters['num_epochs'] = 30
    parameters['lr'] = 1e-7
    parameters['pt_lr'] = 1e-7
    parameters['encoder_size'] = 65803  # the size of gneotyped_p1e-1
    parameters['pretrain_epochs'] = 50
    parameters['num_phenos'] = 3
    parameters['use_phenos'] = True
    parameters['output_type'] = 'tok'
    parameters['embed_dim'] = 64
    parameters['num_heads'] = 4
    parameters['num_layers'] = 4
    parameters['linformer_k'] = 64
    parameters['use_linformer'] = True
    parameters['input_filtering'] = 'random_test_verify'
    return parameters

def get_or_train_net(parameters, train_ids):
    pretrain_snv_toks = get_pretrain_dataset(train_ids, parameters)
    net_file = saved_nets_dir + get_net_savename(parameters)
    if os.path.exists(net_file):
        print("reloading file: {}".format(net_file))
        net = snp_network.get_transformer(parameters, pretrain_snv_toks)
        net = nn.DataParallel(net, environ.use_device_ids)
        net.load_state_dict(torch.load(net_file))
        net = net.to(environ.use_device_ids[0])
    else:
        print("Training new net, no saved net in file '{}'".format(net_file))
        net = snp_network.train_everything(parameters)
        torch.save(net.state_dict(), net_file)
    return net, pretrain_snv_toks


# v6 encoding, pretraining
def combined_v6():
    parameters = get_v6_params()
    train_ids, train, test_ids, test, verify_ids, verify, geno, pheno, enc_ver = get_data(parameters)
    net_file = saved_nets_dir + get_net_savename(parameters)
    net, pretrain_snv_toks = get_or_train_net(parameters, train_ids)
    print("summarising test-set results")
    summarise_net(net, test, parameters, net_file)


def geno_only_pretrain():
    parameters = get_geno_only_params()
    parameters['pretrain_epochs'] = 50
    # TODO loading all this here is overkill
    train_ids, train, test_ids, test, verify_ids, verify, geno, pheno, enc_ver = get_data(parameters)
    pretrain_snv_toks = get_pretrain_dataset(train_ids, parameters)

    # reload trained network if it exists
    net_file = saved_nets_dir + get_net_savename(parameters)
    net, pretrain_snv_toks = get_or_train_net(parameters, train_ids)

    print("summarising test-set results")
    summarise_net(net, test, parameters, net_file)


def geno_only():
    parameters = get_geno_only_params()
    # TODO loading all this here is overkill
    train_ids, train, test_ids, test, verify_ids, verify, geno, pheno, enc_ver = get_data(parameters)

    net, pretrain_snv_toks = get_or_train_net(parameters, train_ids)

    print("summarising test-set results")
    net_file = saved_nets_dir + get_net_savename(parameters)
    summarise_net(net, test, parameters, net_file)

def get_pheno_ternary_parameters():
    parameters = snp_network.default_parameters
    parameters['pretrain_base'] = 'all_unimputed_combined'
    parameters['plink_base'] = 'genotyped_p1e-1'
    parameters['continue_training'] = False
    parameters['train_new_encoder'] = False # re-use encoder if one has been trained
    parameters['encoding_version'] = 5
    parameters['test_frac'] = 0.25
    parameters['verify_frac'] = 0.05
    parameters['batch_size'] = 10
    parameters['num_epochs'] = 100
    parameters['lr'] = 1e-7
    parameters['pt_lr'] = 1e-7
    parameters['encoder_size'] = 65803  # the size of gneotyped_p1e-1
    parameters['pretrain_epochs'] = 50
    parameters['num_phenos'] = 3
    parameters['use_phenos'] = True
    parameters['output_type'] = 'tok'
    parameters['embed_dim'] = 64
    parameters['num_heads'] = 4
    parameters['num_layers'] = 4
    parameters['linformer_k'] = 64
    parameters['use_linformer'] = True
    #parameters['input_filtering'] = 'match_all_phenotypes'
    parameters['input_filtering'] = 'random_test_verify'
    return parameters


def pheno_ternary():
    parameters = get_pheno_ternary_parameters()
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
        net = nn.DataParallel(net, environ.use_device_ids)
        net.load_state_dict(torch.load(net_file))
        net = net.to(environ.use_device_ids[0])
    else:
        print("Training new net, no saved net in file '{}'".format(net_file))
        net = snp_network.train_everything(parameters)

    print("summarising test-set results")
    summarise_net(net, test, parameters, net_file)


def pheno_96():
    parameters = snp_network.default_parameters
    parameters['pretrain_base'] = 'genotyped_p1e-1'
    parameters['plink_base'] = 'genotyped_p1e-1'
    parameters['continue_training'] = False
    parameters['train_new_encoder'] = True
    parameters['encoding_version'] = 2
    parameters['test_frac'] = 0.25
    parameters['verify_frac'] = 0.05
    parameters['batch_size'] = 5
    parameters['num_epochs'] = 60
    parameters['lr'] = 1e-7
    parameters['pt_lr'] = 1e-7
    parameters['encoder_size'] = 65803  # the size of gneotyped_p1e-1
    parameters['pretrain_epochs'] = 0  # original version has no pre-training
    parameters['num_phenos'] = 3
    parameters['use_phenos'] = True
    parameters['output_type'] = 'tok'
    parameters['embed_dim'] = 96
    parameters['num_heads'] = 6
    parameters['num_layers'] = 4
    parameters['linformer_k'] = 96
    parameters['use_linformer'] = True

    # TODO loading all this here is overkill
    train_ids, train, test_ids, test, verify_ids, verify, geno, pheno, enc_ver = get_data(parameters)
    pretrain_snv_toks = get_pretrain_dataset(train_ids, parameters)

    # reload trained network if it exists
    net_file = saved_nets_dir + get_net_savename(parameters)
    net, pretrain_snv_toks = get_or_train_net(parameters, train_ids)

    print("summarising test-set results")
    summarise_net(net, test, parameters, net_file)

def pheno_v1():
    parameters = snp_network.default_parameters
    parameters['pretrain_base'] = 'all_unimputed_combined'
    parameters['plink_base'] = 'genotyped_p1e-1'
    parameters['continue_training'] = False
    parameters['train_new_encoder'] = True
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
    net, pretrain_snv_toks = get_or_train_net(parameters, train_ids)

    print("summarising test-set results")
    summarise_net(net, test, parameters, net_file)

from logistic_regression import remove_bmi

def vs_tin_net():
    parameters = snp_network.default_parameters
    parameters['num_epochs'] = 10
    parameters['pretrain_epochs'] = 0  # original version has no pre-training
    parameters['num_phenos'] = 3
    parameters['pt_lr'] = 1e-7
    parameters['lr'] = 1e-4
    parameters['encoding_version'] = 5
    parameters['encoder_size'] = 123
    parameters['batch_size'] = 256
    parameters['pretrain_base'] = 'tin_fixed_order'
    parameters['plink_base'] = 'tin_fixed_order'
    train_ids, train, test_ids, test, verify_ids, verify, geno, pheno, enc_ver = get_data(parameters)
    net, pretrain_snv_toks = get_or_train_net(parameters, train_ids)
    print("summarising test-set results")
    net_file = saved_nets_dir + get_net_savename(parameters)
    summarise_net(net, test, parameters, net_file)


if __name__ == "__main__":
    home_dir = os.environ.get("HOME")
    os.chdir(home_dir + "/work/gout-transformer")

    vs_tin_net()
    # geno_only()
    # pheno_v1() # done
    # pheno_ternary()
    # combined_v6()
    # pheno_only()
    # pheno_96()
