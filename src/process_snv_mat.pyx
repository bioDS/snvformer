import torch
import numpy as np
import math
from cython.parallel import prange
from tqdm import tqdm
cimport cython


def enc_v6(string_to_tok, tok_to_string, pos: int, p: int, a0, a1):
    a0_toks = np.zeros(p, dtype=np.int32)
    a1_toks = np.zeros(p, dtype=np.int32)
    a01_toks = np.zeros(p, dtype=np.int32)

    cdef int a_tok
    cdef int b_tok
    cdef int ab_tok
    cdef int diff_len

    for i,(a,b) in enumerate(zip(a0,a1)):
        a = str(a)
        b = str(b)
        a_len = len(a)
        b_len = len(b)
        # a, b are both len 1
        if a_len == 1 and b_len == 1:
            a_string = a
            b_string = b
            ab_string = a + ',' + b
        # a > b
        elif a_len > b_len:
            a_string = 'longer'
            b_string = 'shorter'
            ab_string = 'mixed_indel'
        # b > a
        elif b_len > a_len:
            b_string = 'longer'
            a_string = 'shorter'
            ab_string = 'mixed_indel'
        # a, b equal length, both > 1
        elif a_len == b_len and a_len > 1:
            a_string = 'long_sub'
            b_string = 'long_sub'
            ab_string = 'mixed_long_sub'

        # a0
        if a_string in string_to_tok:
            a_tok = string_to_tok[a_string]
        else:
            a_tok = pos
            string_to_tok[a_string] = pos
            tok_to_string[pos] = a_string
            pos = pos + 1
        # a1
        if b_string in string_to_tok:
            b_tok = string_to_tok[b_string]
        else:
            b_tok = pos
            string_to_tok[b_string] = pos
            tok_to_string[pos] = b_string
            pos = pos + 1
        # a01
        if ab_string in string_to_tok:
            ab_tok = string_to_tok[ab_string]
        else:
            ab_tok = pos
            string_to_tok[ab_string] = pos
            tok_to_string[pos] = ab_string
            pos = pos + 1

        a01_toks[i] = ab_tok
        a0_toks[i] = a_tok
        a1_toks[i] = b_tok

    return a0_toks, a1_toks, a01_toks, tok_to_string, string_to_tok


# Just directly uses the snp major/minor 0/1/2 situation.
def enc_v5(string_to_tok, tok_to_string, pos, p: int, a0, a1):
    a0_toks  = np.array(np.repeat(0, len(a0)), dtype=np.int32)
    a01_toks = np.array(np.repeat(1, len(a0)), dtype=np.int32)
    a1_toks  = np.array(np.repeat(2, len(a0)), dtype=np.int32)
    for n in [0, 1, 2]:
        tok_to_string[n] = str(n)
        string_to_tok[str(n)] = n
    return a0_toks, a1_toks, a01_toks, tok_to_string, string_to_tok


def enc_v4(string_to_tok, tok_to_string, pos, p: int, a0, a1):
    a0_toks = np.zeros(p, dtype=np.int32)
    a1_toks = np.zeros(p, dtype=np.int32)
    a01_toks = np.zeros(p, dtype=np.int32)
    diff_lens= np.zeros(p, dtype=np.int32)

    cdef int a_tok
    cdef int b_tok
    cdef int ab_tok
    cdef int diff_len

    for i,(a,b) in enumerate(zip(a0,a1)):
        a = str(a)
        b = str(b)
        a_len = len(a)
        b_len = len(b)
        if (a_len > 1):
            a = 'seq'
        if (b_len > 1):
            b = 'seq'
        ab_string = a + ',' + b

        a_string = a
        b_string = b

        # a0
        if a_string in string_to_tok:
            a_tok = string_to_tok[a_string]
        else:
            a_tok = pos
            string_to_tok[a_string] = pos
            tok_to_string[pos] = a_string
            pos = pos + 1
        # a1
        if b_string in string_to_tok:
            b_tok = string_to_tok[b_string]
        else:
            b_tok = pos
            string_to_tok[b_string] = pos
            tok_to_string[pos] = b_string
            pos = pos + 1
        # a01
        if ab_string in string_to_tok:
            ab_tok = string_to_tok[ab_string]
        else:
            ab_tok = pos
            string_to_tok[ab_string] = pos
            tok_to_string[pos] = ab_string
            pos = pos + 1

        a01_toks[i] = ab_tok
        a0_toks[i] = a_tok
        a1_toks[i] = b_tok
        diff_lens[i] = b_len - a_len

    return a0_toks, a1_toks, a01_toks, tok_to_string, string_to_tok


def enc_v3(string_to_tok, tok_to_string, pos, p: int, a0, a1):
    a0_toks = np.zeros(p, dtype=np.int32)
    a1_toks = np.zeros(p, dtype=np.int32)
    a01_toks = np.zeros(p, dtype=np.int32)

    cdef int a_tok
    cdef int b_tok
    cdef int ab_tok

    for i,(a,b) in enumerate(zip(a0,a1)):
        a = str(a)
        b = str(b)
        a_len = len(a)
        b_len = len(b)
        if (a_len > 1):
            a = a[0] + 'I'
        if (b_len > 1):
            b = b[0] + 'I'
        if b_len >= a_len:
            ab_string = a + ',' + b
        elif b_len < a_len:
            ab_string = a + ',' + 'del'

        a_string = a
        b_string = b

        # a0
        if a_string in string_to_tok:
            a_tok = string_to_tok[a_string]
        else:
            a_tok = pos
            string_to_tok[a_string] = pos
            tok_to_string[pos] = a_string
            pos = pos + 1
        # a1
        if b_string in string_to_tok:
            b_tok = string_to_tok[b_string]
        else:
            b_tok = pos
            string_to_tok[b_string] = pos
            tok_to_string[pos] = b_string
            pos = pos + 1
        # a01
        if ab_string in string_to_tok:
            ab_tok = string_to_tok[ab_string]
        else:
            ab_tok = pos
            string_to_tok[ab_string] = pos
            tok_to_string[pos] = ab_string
            pos = pos + 1

        a01_toks[i] = ab_tok
        a0_toks[i] = a_tok
        a1_toks[i] = b_tok

    return a0_toks, a1_toks, a01_toks, tok_to_string, string_to_tok


# >1 seqs become "XI"
# minor alleles shorter become 'del'
# minor alleles longer become 'ins'
def enc_v2(string_to_tok, tok_to_string, pos, p: int, a0, a1):
    a0_toks = np.zeros(p, dtype=np.int32)
    a1_toks = np.zeros(p, dtype=np.int32)
    a01_toks = np.zeros(p, dtype=np.int32)
    for i,string in enumerate(a0):
        if (len(string) > 1):
            string = string[0] + 'I'
        if string in string_to_tok:
            tok = string_to_tok[string]
        else:
            tok = pos
            string_to_tok[string] = pos
            tok_to_string[pos] = string
            pos = pos + 1
        a0_toks[i] = tok
    for i in range(len(a1)):
        if len(a1[i]) == len(a0[i]):
            if (len(a1[i]) > 1):
                string = a1[i][0] + 'I'
            else:
                string = a1[i]
        elif len(a1[i]) > len(a0[i]):
            string = 'ins'
        elif len(a1[i]) < len(a0[i]):
            string = 'del'
        if string in string_to_tok:
            tok = string_to_tok[string]
        else:
            tok = pos
            string_to_tok[string] = pos
            tok_to_string[pos] = string
            pos = pos + 1
        a1_toks[i] = tok
    for i,(a,b) in enumerate(zip(a0,a1)):
        a = str(a)
        b = str(b)
        a_len = len(a)
        b_len = len(b)
        if (a_len > 1):
            a = a[0] + 'I'
        if (b_len > 1):
            b = b[0] + 'I'
        if b_len == a_len:
            string = a + ',' + b
        elif b_len > a_len:
            string = a + ',' + 'ins'
        elif b_len < a_len:
            string = a + ',' + 'del'
        if string in string_to_tok:
            tok = string_to_tok[string]
        else:
            tok = pos
            string_to_tok[string] = pos
            tok_to_string[pos] = string
            pos = pos + 1
        a01_toks[i] = tok
    return a0_toks, a1_toks, a01_toks, tok_to_string, string_to_tok

def enc_v1(string_to_tok, tok_to_string, pos, p: int, a0, a1):
    a0_toks = np.zeros(p, dtype=np.int32)
    a1_toks = np.zeros(p, dtype=np.int32)
    a01_toks = np.zeros(p, dtype=np.int32)
    for i,string in enumerate(a0):
        if string in string_to_tok:
            tok = string_to_tok[string]
        else:
            tok = pos
            string_to_tok[string] = pos
            tok_to_string[pos] = string
            pos = pos + 1
        a0_toks[i] = tok
    for i in range(len(a1)):
        string = a1[i]
        if string in string_to_tok:
            tok = string_to_tok[string]
        else:
            tok = pos
            string_to_tok[string] = pos
            tok_to_string[pos] = string
            pos = pos + 1
        a1_toks[i] = tok
    for i,(a,b) in enumerate(zip(a0,a1)):
        string = str(a) + ',' + str(b)
        if string in string_to_tok:
            tok = string_to_tok[string]
        else:
            tok = pos
            string_to_tok[string] = pos
            tok_to_string[pos] = string
            pos = pos + 1
        a01_toks[i] = tok
    return a0_toks, a1_toks, a01_toks, tok_to_string, string_to_tok

# val == 0 means we have two 'a0' vals
# val == 2 means two 'a1' vals
# val == 1 means one of each
# def get_tok_mat(a0, a1, vals):
# positions are in GRCh37
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
# 'ai_encoding' means ACTCTT -> AI, 'ins', 'del'
def get_tok_mat(geno, encoding: int = 2):
    a0 = geno.a0.values
    a1 = geno.a1.values
    # tok_mat = torch.zeros(np.shape(vals), dtype=torch.int32)
    encoded_tok_list = []
    string_to_tok = {}
    tok_to_string = {}
    (n_tmp, p_tmp) = geno.shape
    cdef int n = n_tmp
    cdef int p = p_tmp
    cdef int pos = 0

    for special_tok in ['nan', 'ins', 'del', 'cls', 'mask']:
        string_to_tok[special_tok] = pos
        tok_to_string[pos] = special_tok
        pos = pos + 1
    cdef int nan_tok = string_to_tok['nan']

    cdef unsigned char tok

    pos = len(string_to_tok)
    print("identifying tokens")

    diff_lens = None

    if (encoding == 1):
        a0_toks, a1_toks, a01_toks, tok_to_string, string_to_tok = enc_v1(string_to_tok, tok_to_string, pos, p, a0, a1)
    elif (encoding == 2):
        a0_toks, a1_toks, a01_toks, tok_to_string, string_to_tok = enc_v2(string_to_tok, tok_to_string, pos, p, a0, a1)
    elif (encoding == 3):
        a0_toks, a1_toks, a01_toks, tok_to_string, string_to_tok = enc_v3(string_to_tok, tok_to_string, pos, p, a0, a1)
    elif (encoding == 4):
        a0_toks, a1_toks, a01_toks, tok_to_string, string_to_tok, diff_lens = enc_v4(string_to_tok, tok_to_string, pos, p, a0, a1)
    elif (encoding == 5):
        a0_toks, a1_toks, a01_toks, tok_to_string, string_to_tok = enc_v5(string_to_tok, tok_to_string, pos, p, a0, a1)
    elif (encoding == 6):
        a0_toks, a1_toks, a01_toks, tok_to_string, string_to_tok = enc_v6(string_to_tok, tok_to_string, pos, p, a0, a1)

    # geno_mat = np.matrix(geno.values, dtype=np.int32)
    # we can get away with this because there are very few unique variations
    tok_mat = np.zeros((n, p), dtype=np.uint8)
    alleles_differ_mat = np.zeros((n,p), dtype=np.bool_) # N.B. stored as a byte.
    is_nonref_mat = np.zeros((n,p), dtype=np.bool_) # N.B. stored as a byte.

    # memory views for numpy arrays
    cdef int [:] a0_toks_view = a0_toks
    cdef int [:] a1_toks_view = a1_toks
    cdef int [:] a01_toks_view = a01_toks
    cdef unsigned char [:,:] alleles_differ_mat_view = alleles_differ_mat
    cdef unsigned char [:,:] is_nonref_mat_view = is_nonref_mat
    cdef unsigned char [:,:] tok_mat_view = tok_mat
    # cdef int [:,:] geno_mat_view = geno_mat

    print("building token matrix v-{}".format(encoding))

    cdef int batch_size = 1024
    cdef Py_ssize_t ri, ind
    cdef int val
    cdef int [:,:] geno_mat_view
    cdef int actual_row = 0
    cdef int batch
    cdef int enc_ver = encoding
    cdef int actual_batch_len
    for batch in tqdm(range(int(np.ceil(float(n)/batch_size)))):
        geno_mat = np.array(geno[batch*batch_size:(batch+1)*batch_size].values, dtype=np.int32)
        geno_mat_view = geno_mat
        actual_batch_len = geno_mat.shape[0]
        with nogil:
            for ri in prange(actual_batch_len):
                actual_row = batch * batch_size + ri
                if actual_row < n:
                    for ind in range(p):
                        if (enc_ver == 4):
                            val = geno_mat_view[ri, ind]
                            if val == 0:
                                tok = a0_toks_view[ind]
                                is_nonref_mat_view[actual_row, ind] = 0
                                alleles_differ_mat_view[actual_row, ind] = 0
                            elif val == 2:
                                tok = a1_toks_view[ind]
                                is_nonref_mat_view[actual_row, ind] = 1
                                alleles_differ_mat_view[actual_row, ind] = 0
                            elif val == 1:
                                tok = a01_toks_view[ind]
                                is_nonref_mat_view[actual_row, ind] = 1
                                alleles_differ_mat_view[actual_row, ind] = 1
                            else:
                                tok = nan_tok
                        else:
                            val = geno_mat_view[ri, ind]
                            if val == 0:
                                tok = a0_toks_view[ind]
                            elif val == 2:
                                tok = a1_toks_view[ind]
                            elif val == 1:
                                tok = a01_toks_view[ind]
                            else:
                                tok = nan_tok
                        tok_mat_view[actual_row, ind] = tok

    tok_mat = torch.from_numpy(tok_mat)
    if (encoding == 4):
        return tok_mat, is_nonref_mat, alleles_differ_mat, diff_lens, tok_to_string, string_to_tok, len(string_to_tok)
    else :
        return tok_mat, tok_to_string, string_to_tok, len(string_to_tok)
