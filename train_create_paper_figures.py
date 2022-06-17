#!/usr/bin/env python3
import snp_network
import os


def main():

    home_dir = os.environ.get("HOME")
    os.chdir(home_dir + "/work/gout-transformer")

    parameters = snp_network.default_parameters
    parameters['pretrain_base'] = 'all_unimputed_combined'
    parameters['plink_base'] = 'genotyped_p1e-1'
    parameters['ukbb_data'] = '/data/ukbb/'
    parameters['pheno_file'] = 'phenos.csv'
    parameters['continue_training'] = False
    parameters['train_new_encoder'] = True
    parameters['use_device_ids'] = [4]
    parameters['encoding_version'] = 2
    parameters['test_frac'] = 0.25
    parameters['verify_frac'] = 0.05
    parameters['batch_size'] = 5
    parameters['num_epochs'] = 100
    parameters['lr'] = 1e-7
    parameters['pt_lr'] = 1e-7
    parameters['encoder_size'] = -1
    parameters['pretrain_epochs'] = 0  # original version has no pre-training

    snp_network.train_everything(parameters)


if __name__ == "__main__":
    main()
