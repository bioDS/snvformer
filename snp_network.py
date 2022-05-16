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

plink_base = os.environ['PLINK_FILE']
cache_dir = "./cache/"

class simple_mlp(nn.Module):
    def __init__(self, in_size, num_hiddens, depth, vocab_size, embed_dim, max_seq_pos, device) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_dim, device=device)
        self.pos_encoding = ExplicitPositionalEncoding(embed_dim, max_len=max_seq_pos+1)
        self.input_layer = nn.Linear(in_size*embed_dim, num_hiddens, device=device)
        self.blocks = []
        for _ in range(depth):
            self.blocks.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.LayerNorm(num_hiddens, device=device),
                    nn.Linear(num_hiddens, num_hiddens, device=device),
                )
            )
        self.output_layer = nn.Sequential(
            nn.Linear(num_hiddens, 2, device=device), nn.Softmax(-1)
        )
        self.device = device

    def forward(self, x, pos):
        x = x.to(self.device)
        x = self.embedding(x.t()).swapaxes(0,1)
        x = self.pos_encoding(x, pos)
        x = x.reshape(x.shape[0],x.shape[1]*x.shape[2])
        tmp = self.input_layer(x)
        for block in self.blocks:
            tmp = block(tmp)
        tmp = self.output_layer(tmp)
        return tmp


    # net = get_transformer(geno.tok_mat.shape[1], max_seq_pos, geno.num_toks, batch_size, device) #TODO: maybe positions go too high?
    # net = get_mlp(geno.tok_mat.shape[1], geno.num_toks, max_seq_pos, device)
def get_mlp(in_size, vocab_size, max_seq_pos, device):
    num_hiddens = 32
    depth = 3
    embed_dim = 32
    net = simple_mlp(in_size, num_hiddens, depth, vocab_size, embed_dim, max_seq_pos, device)
    return net


def get_transformer(seq_len, num_phenos, max_seq_pos, vocab_size, batch_size, device, output):
    net = TransformerModel(
        seq_len,
        num_phenos,
        max_seq_pos,
        embed_dim=64,
        num_heads=4,
        num_layers=4,
        vocab_size=vocab_size,
        batch_size=batch_size,
        device=device,
        output_type=output,
        use_linformer=True,
        linformer_k=64,
    )
    return net

def get_train_test_simple(geno, pheno, test_split):
    pass

def get_train_test(input_phenos, geno, pheno, test_split):
    # urate = pheno["urate"].values
    gout = pheno["gout"].values
    # test_cutoff = (int)(math.ceil(test_split * geno.tok_mat.shape[0]))

    test_num_gout = (int)(math.ceil(np.sum(gout)*test_split))
    train_num_gout = np.sum(gout) - test_num_gout
    train_gout_pos = np.random.choice(np.where(gout)[0], train_num_gout, replace=False)
    test_gout_pos = np.setdiff1d(np.where(gout), train_gout_pos)
    train_gout_pos.sort()
    test_gout_pos.sort()

    train_control_pos = np.random.choice(np.where(gout == False)[0], train_num_gout, replace=False)
    test_control_pos = np.random.choice(np.setdiff1d(np.where(gout == False)[0], train_control_pos), test_num_gout, replace=False)
    train_control_pos.sort()
    test_control_pos.sort()

    train_all_pos = np.concatenate([train_control_pos, train_gout_pos])
    test_all_pos = np.concatenate([test_control_pos, test_gout_pos])

    positions = tensor(geno.positions)
    test_seqs = geno.tok_mat[test_all_pos,]
    test_phes = gout[test_all_pos]
    train_seqs = geno.tok_mat[train_all_pos]
    train_phes = gout[train_all_pos]
    print("seqs shape: ", train_seqs.shape)
    print("phes shape: ", train_phes.shape)

    age = input_phenos["age"]
    # 0 == Male, 1 == Female
    sex = torch.tensor(input_phenos["sex"].values == "Female", dtype=torch.int64)
    input_pheno_vec = torch.tensor([
        age.values,
        sex,
    ], dtype=torch.int64).t()
    train_pheno_vec = input_pheno_vec[train_all_pos,]
    test_pheno_vec = input_pheno_vec[test_all_pos,]
    # training_dataset = data.TensorDataset(
    #     tensor(train_seqs, device=device), tensor(train_phes, dtype=torch.int64)
    # )
    # test_dataset = data.TensorDataset(
    #     tensor(test_seqs, device=device), tensor(test_phes, dtype=torch.int64)
    # )
    training_dataset = data.TensorDataset(
        train_pheno_vec, positions.repeat(len(train_seqs), 1), tensor(train_seqs), tensor(train_phes, dtype=torch.int64)
    )
    test_dataset = data.TensorDataset(
        test_pheno_vec, positions.repeat(len(test_seqs), 1), tensor(test_seqs), tensor(test_phes, dtype=torch.int64)
    )
    return training_dataset, test_dataset


def train_net(
    net, training_dataset, test_dataset, batch_size, num_epochs, device, learning_rate, prev_num_epochs, test_split
):
    print("beginning training")
    training_iter = data.DataLoader(training_dataset, batch_size, shuffle=True)
    test_iter = data.DataLoader(test_dataset, (int)(batch_size), shuffle=True)
    # trainer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    # trainer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # trainer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
    trainer = torch.optim.AdamW(net.parameters(), lr=learning_rate, amsgrad=True) #TODO worth a try.

    loss = nn.CrossEntropyLoss()

    print("starting training")
    for e in range(num_epochs):
        # print("epoch {}".format(e))
        sum_loss = 0.0
        for phenos, pos, X, y in training_iter:
            # X = X.t()
            X = X.to(device)
            y = y.to(device)
            pos = pos.to(device)
            Yh = net(phenos, X, pos)  # two-value softmax (binary classification)
            l = loss(Yh, y)
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()
            sum_loss += l.mean()
        # if e % 1 == 0:
        if e % (num_epochs / 10) == 0:
            print(
                "epoch {}, mean loss {:.3}, ".format(
                    e, sum_loss / len(training_iter)),
                end="",
            )
            test_loss = 0.0
            with torch.no_grad():
                for phenos, pos, X, y in test_iter:
                    # X = X.t()
                    X = X.to(device)
                    y = y.to(device)
                    pos = pos.to(device)
                    Yh = net(phenos, X, pos)  # two-value softmax (binary classification)
                    l = loss(Yh, y)
                    test_loss += l.mean()
            print("{:.3} (test)".format(test_loss / len(test_iter)))
            print("random few train cases:")
            for phenos, pos, tX, tY in data.Subset(training_dataset, np.random.choice(len(training_dataset), size=5, replace=False)):
                tX = tX.to(device).unsqueeze(0)
                tY = tY.to(device).unsqueeze(0)
                pos = pos.to(device).unsqueeze(0)
                phenos = phenos.to(device).unsqueeze(0)
                tYh = net(phenos, tX, pos)
                tzip = zip(tYh, tY)
                for a, b in tzip:
                    print("{:.3f}, \t {}".format(a[1].item(), b.item()))
            
            net_file = "net_epochs/{}_batch-{}_epoch-{}_test_split-{}_net.pickle".format(
                plink_base, batch_size, e + prev_num_epochs, test_split
            )
            torch.save(net.state_dict(), net_file)

    print("random few train cases:")
    for phenos, pos, tX, tY in data.Subset(training_dataset, np.random.choice(len(training_dataset), size=10, replace=False)):
        tX = tX.to(device).unsqueeze(0)
        tY = tY.to(device).unsqueeze(0)
        pos = pos.to(device).unsqueeze(0)
        phenos = phenos.to(device).unsqueeze(0)
        tYh = net(phenos, tX, pos)
        tzip = zip(tYh, tY)
        for a, b in tzip:
            print("{:.3f}, \t {}".format(a[1].item(), b.item()))

    print("random few test cases:")
    for phenos, pos, tX, tY in data.Subset(test_dataset, np.random.choice(len(test_dataset), size=10, replace=False)):
        tX = tX.to(device).unsqueeze(0)
        tY = tY.to(device).unsqueeze(0)
        pos = pos.to(device).unsqueeze(0)
        phenos = phenos.to(device).unsqueeze(0)
        tYh = net(phenos, tX, pos)
        tzip = zip(tYh, tY)
        for a, b in tzip:
            print("{:.3f}, \t {}".format(a[1].item(), b.item()))

    # check binary accuracy on test set:
    test_total_correct = 0.0
    test_total_incorrect = 0.0
    with torch.no_grad():
        for phenos, pos, tX, tY in test_iter:
            pos = pos.to(device)
            phenos = phenos.to(device)
            tX = tX.to(device)
            tYh = net(phenos, tX, pos)
            binary_tYh = tYh[:, 1] > 0.5
            binary_tY = tY > 0.5
            binary_tY = binary_tY.to(device)
            correct = binary_tYh == binary_tY
            num_correct = torch.sum(correct)
            test_total_correct += num_correct
            test_total_incorrect += len(tX) - num_correct
    train_total_correct = 0.0
    train_total_incorrect = 0.0
    with torch.no_grad():
        for phenos, pos, tX, tY in training_iter:
            tX = tX.to(device)
            pos = pos.to(device)
            phenos = phenos.to(device)
            tYh = net(phenos, tX, pos)
            binary_tYh = tYh[:, 1] > 0.5
            binary_tY = tY > 0.5
            binary_tY = binary_tY.to(device)
            correct = binary_tYh == binary_tY
            num_correct = torch.sum(correct)
            train_total_correct += num_correct
            train_total_incorrect += len(tX) - num_correct
    print(
        "fraction correct: {:.3f} (train), {:.3f} (test)".format(
            train_total_correct /
            (train_total_correct + train_total_incorrect),
            test_total_correct / (test_total_correct + test_total_incorrect),
        )
    )


def reduce_to_half(train):
    train_pos_X = []
    train_pos_y = []
    train_neg_X = []
    train_neg_y = []
    positions, _, _ = train[0]
    for pos, X, y in train:
        if y == 1:
            train_pos_X.append(X.cpu())
            train_pos_y.append(y.cpu())
        else:
            train_neg_X.append(X.cpu())
            train_neg_y.append(y.cpu())

    if len(train_pos_y) < 1:
        raise Exception("must have at least one positive training case")
    neg_training_cases = [i for i in zip(train_neg_X, train_neg_y)]
    num_neg_cases = len(train_neg_y)
    num_pos_cases = len(train_pos_y)
    print("{} pos, {} neg".format(num_pos_cases, num_neg_cases))
    required_neg_cases = num_pos_cases
    # pos_sampler = data.RandomSampler(pos_training_cases, replacement=True, num_samples=required_addition_cases)
    chosen_indices = np.random.choice(num_neg_cases, required_neg_cases, replace=False)
    sampled_neg_cases = [neg_training_cases[i] for i in chosen_indices]
    # sampled_pos_cases = pos_training_cases[[i for i in pos_sampler]]
    new_train_cases = sampled_neg_cases
    print(len(new_train_cases))

    train_seqs = train_pos_X
    # train_seqs.extend(train_neg_X)
    train_phes = train_pos_y
    # train_phes.extend(train_neg_y)
    # train_datasets = []
    for X, y in new_train_cases:
        train_seqs.append(X)
        train_phes.append(y)
        # train_datasets.append(data.TensorDataset(X.expand(0), y.view(1)))
    # train_seqs = tensor(train_seqs)
    train_seqs = torch.stack(train_seqs)
    train_phes = torch.stack(train_phes)
    train_pos = positions.repeat(len(train_seqs), 1)

    training_dataset = data.TensorDataset(
        train_pos, train_seqs, train_phes
    )

    return training_dataset

def amplify_to_half(train):
    train_pos_X = []
    train_pos_y = []
    train_neg_X = []
    train_neg_y = []
    for X, y in train:
        if y == 1:
            train_pos_X.append(X.cpu())
            train_pos_y.append(y.cpu())
        else:
            train_neg_X.append(X.cpu())
            train_neg_y.append(y.cpu())

    if len(train_pos_y) < 1:
        raise Exception("must have at least one positive training case")
    pos_training_cases = [i for i in zip(train_pos_X, train_pos_y)]
    num_neg_cases = len(train_neg_y)
    num_pos_cases = len(train_pos_y)
    print("{} pos, {} neg".format(num_pos_cases, num_neg_cases))
    required_addition_cases = num_neg_cases - num_pos_cases
    # pos_sampler = data.RandomSampler(pos_training_cases, replacement=True, num_samples=required_addition_cases)
    chosen_indices = np.random.choice(num_pos_cases, required_addition_cases)
    sampled_pos_cases = [pos_training_cases[i] for i in chosen_indices]
    # sampled_pos_cases = pos_training_cases[[i for i in pos_sampler]]
    new_train_cases = sampled_pos_cases
    print(len(new_train_cases))

    train_seqs = train_pos_X
    train_seqs.extend(train_neg_X)
    train_phes = train_pos_y
    train_phes.extend(train_neg_y)
    # train_datasets = []
    for X, y in new_train_cases:
        train_seqs.append(X)
        train_phes.append(y)
        # train_datasets.append(data.TensorDataset(X.expand(0), y.view(1)))
    # train_seqs = tensor(train_seqs)

    training_dataset = data.TensorDataset(
        torch.stack(train_seqs), torch.stack(train_phes)
    )

    return training_dataset


def check_pos_neg_frac(dataset: data.TensorDataset):
    pos, neg, unknown = 0, 0, 0
    for (_, _, _, y) in dataset:
        if y == 1:
            pos = pos + 1
        elif y == 0:
            neg = neg + 1
        else:
            unknown = unknown + 1
    return pos, neg, unknown


def prepend_cls_tok(seqs, cls_tok):
    return np.insert(seqs, 0, cls_tok, axis=1)

def translate_unknown(seqs, tok):
    np.place(seqs, seqs == -9, tok)
    return seqs

def dataset_random_n(set: data.TensorDataset, n: int):
    pos, x, y = set[np.random.choice(len(set), size=n, replace=False)]
    subset = data.TensorDataset(
        pos, x, y
    )
    return subset

def get_data(enc_ver, test_split):
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
        train_phenos, X, Y = read_from_plink(small_set=False, subsample_control=True, encoding=enc_ver)
        print("done, writing to pickle")
        with open(train_phenos_file, "wb") as f:
            pickle.dump(train_phenos, f, pickle.HIGHEST_PROTOCOL)
        with open(X_file, "wb") as f:
            pickle.dump(X, f, pickle.HIGHEST_PROTOCOL)
        with open(Y_file, "wb") as f:
            pickle.dump(Y, f, pickle.HIGHEST_PROTOCOL)
        print("done")

    train, test = get_train_test(train_phenos, X, Y, test_split)
    return train, test, X, Y, enc_ver

def main():
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # if (torch.cuda.device_count() > 1):
    #     device = torch.device('cuda:1')
    use_device_ids=[0,6]
    device = use_device_ids[0]
    home_dir = os.environ.get("HOME")
    os.chdir(home_dir + "/work/gout-transformer")

    test_split = 0.3
    # test_split = 0.05 #TODO: just for testing
    train, test, geno, pheno, enc_ver = get_data(2, test_split)

    batch_size = 60
    num_epochs = 200
    lr = 1e-7
    # output = "tok"
    output = "binary"

    new_epoch = num_epochs
    new_net_name = "{}_encv-{}_batch-{}_epochs-{}_p-{}_n-{}_epoch-{}_test_split-{}_output-{}_net.pickle".format(
        plink_base, str(enc_ver), batch_size, num_epochs,
        geno.tok_mat.shape[1], geno.tok_mat.shape[0], new_epoch,
        test_split, output
    )

    max_seq_pos = geno.positions.max()
    continue_training = True
    # continue_training = False

    num_phenos = 2
    net = get_transformer(geno.tok_mat.shape[1], num_phenos, max_seq_pos, geno.num_toks, batch_size, device, output)
    net = nn.DataParallel(net, use_device_ids)
    if (continue_training):
        prev_epoch = 200
        prev_batch_size = 60
        prev_net_name = "{}_encv-{}_batch-{}_epochs-{}_p-{}_n-{}_epoch-{}_test_split-{}_output-{}_net.pickle".format(
            plink_base, str(enc_ver), prev_batch_size, prev_epoch, geno.tok_mat.shape[1], geno.tok_mat.shape[0], prev_epoch, test_split, output
    )
        net.load_state_dict(torch.load(prev_net_name))
        new_epoch = prev_epoch + num_epochs
        new_net_name = "{}_encv-{}_batch-{}_epochs-{}_p-{}_n-{}_epoch-{}_output-{}_net.pickle".format(
            plink_base, str(enc_ver), batch_size, num_epochs, geno.tok_mat.shape[1], geno.tok_mat.shape[0], new_epoch, output
        )
    else:
        prev_epoch = 0
        with open(new_net_name + "_test.pickle", "wb") as f:
            pickle.dump(test, f, pickle.HIGHEST_PROTOCOL)
        with open(new_net_name + "_train.pickle", "wb") as f:
            pickle.dump(train, f, pickle.HIGHEST_PROTOCOL)
    net = net.to(use_device_ids[0])

    print("train dataset: ", check_pos_neg_frac(train))
    print("test dataset: ", check_pos_neg_frac(test))

    train_net(net, train, test, batch_size, num_epochs, device, lr, prev_epoch, test_split)
    # net = get_mlp(geno.tok_mat.shape[1], geno.num_toks, max_seq_pos, device)

    torch.save(net.state_dict(), new_net_name)

if __name__ == "__main__":
    main()
