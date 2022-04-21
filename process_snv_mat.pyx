import torch
import numpy as np
import math
from cython.parallel import prange
cimport cython


# val == 0 means we have two 'a0' vals
# val == 2 means two 'a1' vals
# val == 1 means one of each
# def get_tok_mat(a0, a1, vals):
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def get_tok_mat(geno):
    a0 = geno.a0.values
    a1 = geno.a1.values
    # tok_mat = torch.zeros(np.shape(vals), dtype=torch.int32)
    encoded_tok_list = []
    string_to_tok = {}
    tok_to_string = {}
    (n_tmp,p_tmp) = geno.shape
    cdef int n = n_tmp
    cdef int p = p_tmp

    string_to_tok['nan'] = 0
    tok_to_string[0] = 'nan'
    cdef int nan_tok = 0

    a0_toks = np.zeros(p, dtype=np.int32)
    a1_toks = np.zeros(p, dtype=np.int32)
    a01_toks = np.zeros(p, dtype=np.int32)

    cdef int tok

    pos = len(string_to_tok)
    print("identifying tokens")
    for i,string in enumerate(a0):
        if string in string_to_tok:
            tok = string_to_tok[string]
        else:
            tok = pos
            string_to_tok[string] = pos
            tok_to_string[pos] = string
            pos = pos + 1
        a0_toks[i] = tok
    for i,string in enumerate(a1):
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