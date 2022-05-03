import torch
import numpy as np
import math
from cython.parallel import prange
cimport cython

def enc_v3(p: int, a0, a1):
    a0_toks = np.zeros(p, dtype=np.int32)
    a1_toks = np.zeros(p, dtype=np.int32)
    a01_toks = np.zeros(p, dtype=np.int32)
    string_to_tok = {}
    tok_to_string = {}

    cdef int a_tok
    cdef int b_tok
    cdef int ab_tok
    cdef int pos = 0

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
def enc_v2(p: int, a0, a1):
    a0_toks = np.zeros(p, dtype=np.int32)
    a1_toks = np.zeros(p, dtype=np.int32)
    a01_toks = np.zeros(p, dtype=np.int32)
    string_to_tok = {}
    tok_to_string = {}
    pos = 0
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

def enc_v1(p: int, a0, a1):
    a0_toks = np.zeros(p, dtype=np.int32)
    a1_toks = np.zeros(p, dtype=np.int32)
    a01_toks = np.zeros(p, dtype=np.int32)
    string_to_tok = {}
    tok_to_string = {}
    pos = 0
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
    (n_tmp,p_tmp) = geno.shape
    cdef int n = n_tmp
    cdef int p = p_tmp

    for special_tok in ['nan', 'ins', 'del']:
        string_to_tok[special_tok] = pos
        tok_to_string[pos] = special_tok
        pos = pos + 1
    cdef int nan_tok = 0


    cdef int tok

    pos = len(string_to_tok)
    print("identifying tokens")

    if (encoding == 1):
        a0_toks, a1_toks, a01_toks, tok_to_string, string_to_tok = enc_v1(p, a0, a1)
    elif (encoding == 2):
        a0_toks, a1_toks, a01_toks, tok_to_string, string_to_tok = enc_v2(p, a0, a1)
    elif (encoding == 3):
        a0_toks, a1_toks, a01_toks, tok_to_string, string_to_tok = enc_v3(p, a0, a1)

    geno_mat = np.matrix(geno.values, dtype=np.int32)
    tok_mat = np.zeros((n, p), dtype=np.int32)

    # memory views for numpy arrays
    cdef int [:] a0_toks_view = a0_toks
    cdef int [:] a1_toks_view = a1_toks
    cdef int [:] a01_toks_view = a01_toks
    cdef int [:,:] tok_mat_view = tok_mat
    cdef int [:,:] geno_mat_view = geno_mat

    print("building token matrix")

    # for ri, row in enumerate(geno_mat):
    cdef Py_ssize_t ri, ind
    cdef int val
    with nogil:
        for ri in prange(n):
            # row = geno_mat[ri,]
            for ind in range(p):
                val = geno_mat_view[ri,ind]
                if val == 0:
                    tok = a0_toks_view[ind]
                elif val == 2:
                    tok = a1_toks_view[ind]
                elif val == 1:
                    tok = a01_toks_view[ind]
                else:
                    tok = nan_tok
                tok_mat_view[ri, ind] = tok
    
    tok_mat = torch.from_numpy(tok_mat)
    return tok_mat, tok_to_string, string_to_tok, len(string_to_tok)