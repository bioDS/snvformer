import torch
import numpy as np
import math

# val == 0 means we have two 'a0' vals
# val == 2 means two 'a1' vals
# val == 1 means one of each
def get_tok_mat(a0, a1, vals):
    tok_mat = torch.zeros(np.shape(vals), dtype=torch.int32)
    string_to_tok = {}
    tok_to_string = {}

    string_to_tok['nan'] = 0
    tok_to_string[0] = 'nan'
    num_toks = 1
    for ri, row in enumerate(vals):
        for ind,val in enumerate(row):
            if math.isnan(val):
                string = 'nan'
            elif val == 0:
                string = a0[ind]
            elif val == 2:
                string = a1[ind]
            elif val == 1:
                string = a0[ind] + ',' + a1[ind]

            if string in string_to_tok:
                tok = string_to_tok[string]
            else:
                tok = num_toks
                num_toks = num_toks + 1
                string_to_tok[string] = tok
                tok_to_string[tok] = string
            tok_mat[ri, ind] = tok
    
    return tok_mat, tok_to_string, string_to_tok, num_toks