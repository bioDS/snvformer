#!/usr/bin/env python3
import torch
from torch import nn
from torch.utils import data
import math
import pickle
from os.path import exists
import numpy as np
from snp_input import get_data, get_pretrain_dataset, Tokenised_SNVs
from self_attention_net import Encoder, TransformerModel, ExplicitPositionalEncoding
from tqdm import tqdm
# from environ import ukbb_data, pheno_file
import environ

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


def get_pretrained_encoder(state_file: str, params, snv_toks):
    print("attempting to load encoder from file {}".format(state_file))
    encoder = get_encoder(params, snv_toks)
    encoder = nn.DataParallel(encoder, environ.use_device_ids)
    encoder.load_state_dict(torch.load(state_file))
    return encoder


def get_encoder_from_params(p: dict, pretrain_snv_toks):
    return get_encoder(p['encoder_size'], p['num_phenos'], pretrain_snv_toks.positions.max(), pretrain_snv_toks.num_toks, p['batch_size'], environ.use_device_ids[0], pretrain_snv_toks.string_to_tok['cls'], p)


def get_encoder(params, pretrain_snv_toks: Tokenised_SNVs):
    cls_tok = pretrain_snv_toks.string_to_tok['cls']
    vocab_size = pretrain_snv_toks.num_toks
    max_seq_pos = pretrain_snv_toks.positions.max()
    encoder = Encoder(
        params,
        cls_tok,
        vocab_size,
        max_seq_pos,
    )
    return encoder


def transformer_from_encoder(encoder, params):
    net = TransformerModel(encoder, params)
    return net


def get_transformer(params, snv_toks):
    net = TransformerModel(
        get_encoder(params, snv_toks), params)
    return net


# mask sequences and positions at random, fill in the gaps
def mask_sequence(seqs, pos, frac, tokenised_snvs: Tokenised_SNVs):
    rng = np.random.default_rng()
    mask_tok = tokenised_snvs.string_to_tok["mask"]
    # ratios are from BERT
    frac_masked = 0.8
    frac_random = 0.1
    frac_pos_masked = 0.1
    # frac_unchanged = 0.1
    mask_positions = []
    random_positions = []
    mask_pos_indices = []

    # a[[0,1],[1,1]] = [7,8]
    # do each batch separately
    for seq in range(seqs.shape[0]):
        pos_inds = rng.choice(seqs.shape[1], size=(int)(math.ceil(frac_pos_masked*seqs.shape[1])), replace=False)
        seq_change_positions = rng.choice(seqs.shape[1], size=(int)(math.ceil(frac*seqs.shape[1])), replace=False)
        seq_mask_positions = [(seq,i) for i in rng.choice(seq_change_positions, (int)(math.ceil(frac_masked*len(seq_change_positions))), replace=False)]
        seq_random_positions = [(seq,i) for i in rng.choice(seq_change_positions, (int)(math.ceil(frac_random*len(seq_change_positions))), replace=False)]
        mask_positions.extend(seq_mask_positions)
        random_positions.extend(seq_random_positions)
        seq_pos_mask_inds = [(seq,i) for i in pos_inds]
        mask_pos_indices.extend(seq_pos_mask_inds)
    # replace mask tokens
    mask_positions = np.transpose(np.array(mask_positions))
    random_positions = np.transpose(np.array(random_positions))
    mask_pos_indices = np.transpose(np.array(mask_pos_indices))
    seqs[mask_positions] = mask_tok
    pos[mask_pos_indices] = -1
    # replace random tokens
    new_random_toks = torch.tensor(rng.choice([i for i in tokenised_snvs.tok_to_string.keys()], random_positions.shape[1], replace=True), dtype=torch.uint8)
    seqs[random_positions] = new_random_toks
    return seqs, pos

def pretrain_encoder(
    encoder, pretrain_snv_toks, batch_size, num_epochs,
        device, learning_rate, pretrain_base, params, pretrain_log_file):

    print("beginning pre-training encoder")
    positions = pretrain_snv_toks.positions
    pos_seq_dataset = data.TensorDataset(positions.repeat(pretrain_snv_toks.tok_mat.shape[0], 1), pretrain_snv_toks.tok_mat)
    training_iter = data.DataLoader(pos_seq_dataset, batch_size, shuffle=True)
    # Nothing should be on the CPU
    for param in encoder.parameters():
        if param.device.type == 'cpu':
            print(param.device)
            print(param.shape)
            break
    trainer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate, amsgrad=True)
    loss = nn.CrossEntropyLoss()

    class_size = encoder.module.embed_dim - encoder.module.pos_size
    rng = np.random.default_rng()

    # loop training set
    for e in range(num_epochs):
        sum_loss = 0.0
        num_steps = 0
        for pos, seqs in tqdm(training_iter, ncols=0):
            #TODO: take this out
            # if (num_steps > 2):
            #     break
            phenos = torch.zeros(seqs.shape[0], encoder.module.num_phenos)
            # take a subset of SNVs the size of the encoder input
            input_size = encoder.module.seq_len
            chosen_positions = rng.choice(seqs.shape[1], input_size, replace=False)
            chosen_positions.sort()
            seqs = seqs[:,chosen_positions]
            pos = pos[:,chosen_positions]
            # randomly mask SNVs (and some positions)
            masked_seqs, masked_pos = mask_sequence(seqs, pos, 0.40, pretrain_snv_toks)

            # get encoded sequence
            phenos = phenos.to(device)
            masked_seqs = masked_seqs.to(device)
            masked_pos = masked_pos.to(device)
            _, _, pred_seqs = encoder(phenos, masked_seqs, masked_pos)

            # remove positions
            # ignore first item in sequence, it's the [cls] token
            pred_class_probs = torch.softmax(pred_seqs[:,:,0:class_size], dim=2)
            pred_class_probs = torch.swapdims(pred_class_probs, 1, 2)
            # pred_classes = torch.argmax(pred_classes_probs, dim=2)
            # true_classes = torch.nn.functional.one_hot(seqs.long()[:,1:], num_classes=pred_class_probs.shape[2])
            true_classes = seqs.long().to(device)

            # penalise mis-prediced values
            l = loss(pred_class_probs, true_classes)
            #TODO: also positions? We give these, so it's a bit unnecessary.

            trainer.zero_grad()
            l.mean().backward()
            trainer.step()
            sum_loss += l.mean()
            num_steps = num_steps + 1
            # if (num_steps > 10):
            #     break


        encoder_file = environ.wip_nets_dir + get_net_savename(params) + ".encoder"
        encoder_file = "net_epochs/pretrain_{}_epoch-{}_encoder_encsize-{}.pickle".format(
            pretrain_base, e, encoder.module.seq_len
        )
        torch.save(encoder.state_dict(), encoder_file)
        s = "epoch {}, loss {}".format(e, sum_loss/num_steps)
        print(s)
        pretrain_log_file.write(s)


def subsample_forward(net: nn.DataParallel, seqs, pos, phenos, device):
    rng = np.random.default_rng()
    input_size = net.module.encoder.seq_len
    chosen_positions = rng.choice(seqs.shape[1], input_size, replace=False)
    chosen_positions.sort()
    seqs = seqs[:, chosen_positions]
    pos = pos[:, chosen_positions]
    phenos = phenos.to(device)
    seqs = seqs.to(device)
    pos = pos.to(device)
    return net(phenos, seqs, pos)


def mask_forward(net, seqs: torch.Tensor, pos, phenos, device, snv_toks: Tokenised_SNVs):
    encoder_size = net.module.encoder.seq_len
    mask_tok = snv_toks.string_to_tok['mask']
    orig_seq_len = seqs.shape[1]
    expanded_seqs = seqs.expand(seqs.shape[0], encoder_size)
    expanded_seqs[:, orig_seq_len:] = mask_tok
    expanded_pos = pos[:, encoder_size]
    expanded_pos[:, orig_seq_len:] = -1
    net(phenos, expanded_seqs, expanded_pos)


def process_input(subsample, net, seqs, pos, phenos, device, snv_toks, params):
    if seqs.shape[1] < net.module.encoder.seq_len:
        return mask_forward(net, seqs, pos, phenos, device, snv_toks)
    if subsample:
        return subsample_forward(net, seqs, pos, phenos, device)
    else:
        return net(phenos, seqs, pos)


def train_net(
    net, training_dataset, test_dataset, batch_size, num_epochs,
        device, learning_rate, prev_num_epochs, test_split, train_log_file, plink_base, snv_toks, params, subsample_input=False
):
    training_iter = data.DataLoader(training_dataset, batch_size, shuffle=True)
    test_iter = data.DataLoader(test_dataset, (int)(batch_size), shuffle=True)
    trainer = torch.optim.AdamW(net.parameters(), lr=learning_rate, amsgrad=True)

    loss = nn.CrossEntropyLoss()

    print("starting training")
    for e in range(num_epochs):
        sum_loss = 0.0
        for phenos, pos, X, y in tqdm(training_iter, ncols=0):
            # X = X.t()
            X = X.to(device)
            y = y.to(device)
            pos = pos.to(device)
            # Yh = net(phenos, X, pos)  # two-value softmax (binary classification)
            Yh = process_input(subsample_input, net, X, pos, phenos, device, snv_toks, params)
            l = loss(Yh, y)
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()
            sum_loss += l.mean()
        if e % (num_epochs / 10) == 0:
            test_loss = 0.0
            with torch.no_grad():
                for phenos, pos, X, y in test_iter:
                    X = X.to(device)
                    y = y.to(device)
                    pos = pos.to(device)
                    # Yh = net(phenos, X, pos)  # two-value softmax (binary classification)
                    Yh = process_input(subsample_input, net, X, pos, phenos, device, snv_toks, params)
                    l = loss(Yh, y)
                    test_loss += l.mean()
            tmpstr = "epoch {}, mean loss {:.5}, {:.5} (test)".format(
                    e, sum_loss / len(training_iter), test_loss / len(test_iter))
            print(tmpstr)
            train_log_file.write(tmpstr)
            print("random few train cases:")
            for phenos, pos, tX, tY in data.Subset(training_dataset, np.random.choice(len(training_dataset), size=5, replace=False)):
                tX = tX.to(device).unsqueeze(0)
                tY = tY.to(device).unsqueeze(0)
                pos = pos.to(device).unsqueeze(0)
                phenos = phenos.to(device).unsqueeze(0)
                tYh = process_input(subsample_input, net, tX, pos, phenos, device, snv_toks, params)
                tzip = zip(tYh, tY)
                for a, b in tzip:
                    print("{:.3f}, \t {}".format(a[1].item(), b.item()))

            net_file = "net_epochs/{}_epoch-{}".format(get_net_savename(params), e)
            torch.save(net.state_dict(), net_file)

    print("random few train cases:")
    for phenos, pos, tX, tY in data.Subset(training_dataset, np.random.choice(len(training_dataset), size=10, replace=False)):
        tX = tX.to(device).unsqueeze(0)
        tY = tY.to(device).unsqueeze(0)
        pos = pos.to(device).unsqueeze(0)
        phenos = phenos.to(device).unsqueeze(0)
        tYh = process_input(subsample_input, net, tX, pos, phenos, device, snv_toks, params)
        tzip = zip(tYh, tY)
        for a, b in tzip:
            print("{:.3f}, \t {}".format(a[1].item(), b.item()))

    print("random few test cases:")
    for phenos, pos, tX, tY in data.Subset(test_dataset, np.random.choice(len(test_dataset), size=10, replace=False)):
        tX = tX.to(device).unsqueeze(0)
        tY = tY.to(device).unsqueeze(0)
        pos = pos.to(device).unsqueeze(0)
        phenos = phenos.to(device).unsqueeze(0)
        tYh = process_input(subsample_input, net, tX, pos, phenos, device, snv_toks, params)
        tzip = zip(tYh, tY)
        for a, b in tzip:
            print("{:.3f}, \t {}".format(a[1].item(), b.item()))

    # check binary accuracy on test set:
    test_total_correct = 0.0
    test_total_incorrect = 0.0
    sum_loss = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for phenos, pos, tX, tY in test_iter:
            pos = pos.to(device)
            phenos = phenos.to(device)
            tX = tX.to(device)
            tY = tY.to(device)
            tYh = process_input(subsample_input, net, tX, pos, phenos, device, snv_toks, params)
            binary_tYh = tYh[:, 1] > 0.5
            binary_tY = tY > 0.5
            binary_tY = binary_tY.to(device)
            correct = binary_tYh == binary_tY
            num_correct = torch.sum(correct)
            test_total_correct += num_correct
            test_total_incorrect += len(tX) - num_correct
            l = loss(tYh, tY)
            test_loss += l.mean()
    train_total_correct = 0.0
    train_total_incorrect = 0.0
    with torch.no_grad():
        for phenos, pos, tX, tY in training_iter:
            tX = tX.to(device)
            tY = tY.to(device)
            pos = pos.to(device)
            phenos = phenos.to(device)
            tYh = process_input(subsample_input, net, tX, pos, phenos, device, snv_toks, params)
            binary_tYh = tYh[:, 1] > 0.5
            binary_tY = tY > 0.5
            binary_tY = binary_tY.to(device)
            correct = binary_tYh == binary_tY
            num_correct = torch.sum(correct)
            train_total_correct += num_correct
            train_total_incorrect += len(tX) - num_correct
            l = loss(tYh, tY)
            sum_loss += l.mean()
    tmpstr = "epoch {}, mean loss {:.3}, {:.3} (test)".format(
            e, sum_loss / len(training_iter), test_loss / len(test_iter))
    print(tmpstr)
    train_log_file.write(tmpstr)
    tmpstr = "final fraction correct: {:.3f} (train), {:.3f} (test)".format(
            train_total_correct /
            (train_total_correct + train_total_incorrect),
            test_total_correct / (test_total_correct + test_total_incorrect),
        )
    print(tmpstr)
    train_log_file.write(tmpstr)


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


# adds the token for 'cls' to the beginning of the sequence, with position '-1'
def prepend_cls_tok(seqs, pos, cls_tok):
    # return np.insert(seqs, 0, cls_tok, axis=1)
    pass


def translate_unknown(seqs, tok):
    np.place(seqs, seqs == -9, tok)
    return seqs


def dataset_random_n(set: data.TensorDataset, n: int):
    pos, x, y = set[np.random.choice(len(set), size=n, replace=False)]
    subset = data.TensorDataset(
        pos, x, y
    )
    return subset


default_parameters = {
    'pretrain_base': 'all_unimputed_combined',
    'plink_base': 'genotyped_p1e-1',
    'continue_training': False,
    'train_new_encoder': False,
    'encoding_version': 2,
    'test_frac': 0.25,
    'verify_frac': 0.05,
    'batch_size': 5,
    'num_epochs': 10,
    'lr': 1e-7,
    'output_type': 'tok',
    'pretrain_epochs': 10,
    'use_phenos': True,
    'embed_dim': 96,
    'num_heads': 4,
    'num_layers': 6,
    'linformer_k': 96,
    'use_linformer': True,
    'input_filtering': 'random_test_verify',
}


def get_net_savename(parameters: dict):
    par_str = "_".join("{}".format(v) for k, v in parameters.items())
    par_str += '.net'
    return par_str.replace('/', '-')


def train_everything(params=default_parameters):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # if (torch.cuda.device_count() > 1):
    #     device = torch.device('cuda:1')
    # load parameters
    pretrain_base = params['pretrain_base']
    plink_base = params['plink_base']
    continue_training = params['continue_training']
    train_new_encoder = params['train_new_encoder']
    encoding_version = params['encoding_version']
    test_frac = params['test_frac']
    verify_frac = params['verify_frac']
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    lr = params['lr']
    output = params['output_type']
    encoder_size = params['encoder_size']
    pt_epochs = params['pretrain_epochs']
    pt_lr = params['pt_lr']
    use_device_ids = environ.use_device_ids

    device = use_device_ids[0]

    train_ids, train, test_ids, test, verify_ids, verify, geno, pheno, enc_ver = get_data(params)
    snv_toks = geno

    net_dir = environ.saved_nets_dir

    new_epoch = num_epochs
    new_net_name = net_dir + "{}_encv-{}_batch-{}_epochs-{}_p-{}_n-{}_epoch-{}_test_split-{}_output-{}_net.pickle".format(
        plink_base, str(enc_ver), batch_size, num_epochs,
        geno.tok_mat.shape[1], geno.tok_mat.shape[0], new_epoch,
        test_frac, output
    )

    # Pre-training
    pretrain_snv_toks = get_pretrain_dataset(train_ids, params)

    print("geno num toks: {}".format(geno.num_toks))
    print("pretrain_snv_toks num toks: {}".format(pretrain_snv_toks.num_toks))
    geno_toks = set([*geno.tok_to_string.keys()])
    pt_toks = set([*pretrain_snv_toks.tok_to_string.keys()])
    ft_new_toks = geno_toks - geno_toks.intersection(pt_toks)
    if (len(ft_new_toks) > 0):
        print([geno.tok_to_string[i] for i in ft_new_toks])
        raise Exception("Warning, fine tuning set contains new tokens!")

    if (encoder_size == -1):
        encoder_size = geno.tok_mat.shape[1]  # the natural size for this input
    print("creating encoder w/ input size: {}".format(encoder_size))
    encoder_params = {k:params[k] for k in('pretrain_base', 'encoding_version', 'test_frac', 'verify_frac', 'batch_size', 'pt_lr', 'encoder_size', 'num_phenos', 'pretrain_epochs', 'input_filtering')}
    encoder_file = environ.saved_nets_dir + get_net_savename(encoder_params) + ".encoder"
    # encoder_file = "pretrained_encoder_input-{}_encv-{}_insize-{}_run_epochs-{}.net".format(
    #     pretrain_base, enc_ver, encoder_size, pt_epochs)
    if (train_new_encoder or not exists(encoder_file)):
        pt_batch_size = batch_size
        encoder = get_encoder(params, pretrain_snv_toks)
        encoder = encoder.cuda(device)
        encoder = nn.DataParallel(encoder, use_device_ids)

        pt_net_name = "bs-{}_epochs-{}_lr-{}_pretrained.net".format(pt_batch_size, pt_epochs, pt_lr)
        pt_log_file = open(pt_net_name + ".log", "w")
        print("pre-training encoder with sequences of length {}".format(pretrain_snv_toks.tok_mat.shape[1]))
        # prepend_cls_tok(pretrain_snv_toks.tok_mat, pretrain_snv_toks.string_to_tok['cls'])
        pretrain_encoder(encoder, pretrain_snv_toks, pt_batch_size, pt_epochs, device, pt_lr, pretrain_base, params, pt_log_file)
        pt_log_file.close()
        torch.save(encoder.state_dict(), encoder_file)
    else:
        encoder = get_pretrained_encoder(encoder_file, params, pretrain_snv_toks)

    # Fine-tuning
    net = transformer_from_encoder(encoder.module, params)

    net = nn.DataParallel(net, use_device_ids)
    if (continue_training):
        prev_params = params['prev_params']
        prev_net_name = environ.saved_nets_dir + get_net_savename(prev_params)
        net.load_state_dict(torch.load(prev_net_name))
        new_net_name = environ.saved_nets_dir + get_net_savename(params)
    else:
        prev_epoch = 0
    net = net.to(use_device_ids[0])

    train_log_file = open(new_net_name + "_log.txt", "w")

    print("train dataset: ", check_pos_neg_frac(train))
    print("test dataset: ", check_pos_neg_frac(test))
    # prepend_cls_tok(train, geno.string_to_tok['cls'])
    # prepend_cls_tok(test, geno.string_to_tok['cls'])
    train_net(net, train, test, batch_size, num_epochs, device, lr, prev_epoch, test_frac, train_log_file, plink_base, snv_toks, params, subsample_input=False)

    train_log_file.close()
    # net = get_mlp(geno.tok_mat.shape[1], geno.num_toks, max_seq_pos, device)

    torch.save(net.state_dict(), new_net_name)
    return net

if __name__ == "__main__":
    train_everything()
