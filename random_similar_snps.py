#!/usr/bin/env python3
import numpy as np
from pandas_plink import write_plink1_bin
import torch
from xarray import DataArray
import pandas as pd
import csv

plink_base="random"
bed_file = plink_base+".bed"
bim_file = plink_base+".bim"
fam_file = plink_base+".fam"

num_snps = 500000
num_samples = 10000
positions = np.random.choice(range(10000000), size=num_snps, replace=False)
positions.sort()
snp_names = ["snp" + str(i) for i in positions]

values = np.random.sample((num_samples, num_snps))
variant_names = ["variant" + str(i) for i in range(num_snps)]
sample_ids = [i for i in range(num_samples)]

random_geno = DataArray(values, dims=["sample", "variant"], coords=dict(
    sample = sample_ids,
    fid = ("sample", sample_ids),
    iid = ("sample", sample_ids),
    variant = variant_names,
    snp = ("variant", variant_names),
    chrom = ("variant", ['1' for i in range(num_snps)]),
    pos = ("variant", positions),
    a0 = ("variant", np.random.choice(['A', 'C', 'G', 'T'], num_snps)),
    a1 = ("variant", np.random.choice(['A', 'C', 'G', 'T'], num_snps)),
))

write_plink1_bin(random_geno, bed_file, bim_file, fam_file)

urate = np.random.normal(size=num_samples)
gout = np.random.choice([True,False], num_samples)

urate_df = pd.DataFrame({"eid": sample_ids, "gout": gout, "urate": urate})
urate_df.to_csv("random_urate.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

withdrawn_ids = pd.DataFrame(np.random.choice(sample_ids, 10))
withdrawn_ids.to_csv("w12611_20220222.csv", index=False, header=False)