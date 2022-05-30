#!/usr/bin/env python3
from pyensembl import EnsemblRelease
import numpy as np
import pandas


def get_gene_hit_counts(chromosomes, positions, scores):
    print('summarrising gene hits for {} positions'.format(len(positions)))

    data = EnsemblRelease(60) # 60 is an arbitrary choice that happens to use GRCh37
    gene_names = [data.gene_names_at_locus(chrom, loci) for chrom, loci in zip(chromosomes, positions)]
    all_names = [g for lg in gene_names for g in lg]
    # print(gene_names)
    # print(all_names)
    names, counts = np.unique(all_names, return_counts=True)
    nc_df = pandas.DataFrame({'name': names, 'count': counts})
    nc_df.sort_values(by='count', inplace=True, ascending=False)
    print("most common gene hits:")
    print(nc_df[0:10])

    scores_for_names = {}
    for chrom, loci, score in zip(chromosomes, positions, scores):
        these_names = data.gene_names_at_locus(chrom, loci)
        for name in these_names:
            if (name in scores_for_names):
                scores_for_names[name] += score
            else:
                scores_for_names[name] = score
    scores = pandas.DataFrame({'name': scores_for_names.keys(), 'score': scores_for_names.values()})
    scores.sort_values(by="score", inplace=True, ascending=False)
    print(scores[0:10])

    


# from a run on all_gwas
test_chrom = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3,
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4,
       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5,
       5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6,
       7, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
       11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
       11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
       11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12,
       12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
       12, 12, 13, 14, 15, 15, 15, 15, 15, 15, 15,
       15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16,
       16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
       17, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19,
       19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20,
       20, 20, 20, 22]
test_sig_indices = [ 15876758,  26987646,  27025940,  27046982,  27073924,  27181067,
        27204183,  27248283,  27255904,  27282676, 145831160, 150629152,
       150631917, 150649180, 150650005, 150659829, 150670484, 150679033,
       150711896, 150748786, 150804429, 150908639, 150956128, 154993973,
       155137341, 155184975, 155229310, 163650546, 163706566, 205062433,
       211998157,  27631474,  27722416,  27750997,  27827326,  32563426,
       121318166, 121321830, 148511805, 148527901, 148576646, 148608239,
       148657117, 148669774, 148748577, 148750509, 148755237, 148810157,
       148810549, 148892355, 148937370, 169982422, 170041299, 176765594,
       183101208, 203288774, 211588178, 211601116, 213316881, 213348405,
       242252792, 242302934,  47292827,  48961019,  49766849,  49833029,
        49846041,  50025181,  50039631,  51531688,  52342529,  52619792,
        52677960,  52685305,  52793416,  52803205,  52833881,  52983434,
        53059027,  53061985,  53087482,  53107600,  53169651, 121638382,
       141643694, 141683097, 141732694, 141749775, 141754558, 141785395,
       141786073, 141799690, 141799993, 141805388, 141857052,   9135723,
         9374132,   9375513,   9413299,   9442822,   9452734,   9490919,
         9507035,   9536082,   9555408,   9586764,   9597021,   9631591,
         9678661,   9709993,   9714347,   9729265,   9736033,   9751331,
         9774138,   9795788,   9809859,   9816531,   9842403,   9853869,
         9884876,   9893498,   9946163,   9956079,   9996996,  10016666,
        10022161,  10023939,  10086670,  10094105,  10095422,  10111214,
        10119244,  10121656,  10121994,  10136149,  10139906,  10139913,
        10143400,  10148881,  10149741,  10158961,  10174190,  10185854,
        10196638,  10197663,  10225162,  10237713,  10242730,  10250399,
        10272055,  10326188,  10331887,  10350948,  10377405,  10381719,
        10452986,  10480452,  10511483,  10513072,  10515142,  10516618,
        10595582,  10648422,  10654270,  10683398,  10699629,  10704515,
        10724422,  10729314,  10754359,  10767202,  10768435,  10789519,
        10795046,  10832639,  88427791,  88505945,  88789190,  88839927,
        88843210,  88910425,  88978139,  89074547,  89074808,  89075031,
        89080690,  89092512,  89098305,  89102550,  89108845,  89127082,
        89236028,  89910687,  90040687,  90052439,  90059311, 120431469,
       120450159, 144086245, 144132344, 144152897, 144169712, 144185429,
       144187816,  53320203,  72437054,  72442531,  72447795,  72455422,
        90366216,  90397594, 176825069,  25747939,  25840450,  25918335,
        25953360,  26362119,  27746692,  29918918,  43826681,  97876593,
        60261070,  60264613,  60280911,  60286816,  60348886,  88921718,
        88949406, 126410804, 126413979, 126414084,   2148855,  10243053,
        30432220,  30755111,  30762936,  63878131,  63912189,  63931607,
        64136721,  64391535,  64438711,  64683532,  64711771,  64727721,
        64730318,  65364629,  65435383,  65436792,  65465002,  65493191,
        65641496,  65732800,  65756808,  65870938,  65890597,  69501014,
       119217455, 119217957, 120044762,  57733539,  57754503,  57789147,
        57791986,  57797601,  57824981,  58103047,  58431526,  58438177,
        58445779,  58472363,  78785722, 111854285, 121362330, 122528460,
       122582160, 122606596,  96708108,  96919620,  76061453,  76097412,
        76190044,  76206017,  76237284,  76280307,  76300265,  76330272,
        76461656,  76686527,  99271135,  99273032,  99276297,  99280254,
        99289607,  99305829,  99311861,  25002600,  51714113,  53812783,
        69552512,  69565894,  69855627,  69913996,  69926280,  69967732,
        79702641,  79910843,  89675579,  53360264,  57744576,  57838401,
        57855319,  57884616,   7184238,   7185576,   7219697,  18388250,
        33220101,  33242146,  33298122,  33338118,  33346046,  33346883,
        33362784,  33376203,  33390796,  33406549,  32948753,  33220070,
        33288511,  33288655,  33378733,  33419680,  44331778]

test_scores = [0.0048, 0.0046, 0.0047, 0.0047, 0.0047, 0.0046, 0.0047, 0.0047, 0.0047,
        0.0047, 0.0049, 0.0053, 0.0048, 0.0046, 0.0046, 0.0047, 0.0048, 0.0046,
        0.0047, 0.0047, 0.0050, 0.0046, 0.0047, 0.0048, 0.0049, 0.0046, 0.0051,
        0.0049, 0.0046, 0.0048, 0.0047, 0.0046, 0.0048, 0.0048, 0.0046, 0.0047,
        0.0047, 0.0048, 0.0047, 0.0046, 0.0048, 0.0047, 0.0049, 0.0050, 0.0048,
        0.0046, 0.0048, 0.0046, 0.0049, 0.0047, 0.0047, 0.0046, 0.0048, 0.0048,
        0.0048, 0.0049, 0.0049, 0.0048, 0.0046, 0.0046, 0.0048, 0.0047, 0.0047,
        0.0049, 0.0047, 0.0046, 0.0047, 0.0047, 0.0047, 0.0047, 0.0048, 0.0047,
        0.0048, 0.0050, 0.0048, 0.0050, 0.0049, 0.0046, 0.0046, 0.0049, 0.0046,
        0.0051, 0.0049, 0.0049, 0.0049, 0.0047, 0.0046, 0.0047, 0.0047, 0.0051,
        0.0047, 0.0050, 0.0046, 0.0048, 0.0048, 0.0049, 0.0046, 0.0048, 0.0048,
        0.0047, 0.0047, 0.0046, 0.0047, 0.0047, 0.0047, 0.0048, 0.0048, 0.0050,
        0.0047, 0.0047, 0.0046, 0.0047, 0.0046, 0.0046, 0.0047, 0.0046, 0.0046,
        0.0052, 0.0047, 0.0047, 0.0047, 0.0047, 0.0048, 0.0046, 0.0046, 0.0049,
        0.0048, 0.0047, 0.0052, 0.0048, 0.0046, 0.0050, 0.0047, 0.0047, 0.0048,
        0.0051, 0.0048, 0.0046, 0.0048, 0.0047, 0.0047, 0.0047, 0.0049, 0.0046,
        0.0049, 0.0046, 0.0050, 0.0046, 0.0046, 0.0046, 0.0047, 0.0046, 0.0046,
        0.0050, 0.0052, 0.0050, 0.0047, 0.0047, 0.0047, 0.0046, 0.0047, 0.0047,
        0.0047, 0.0048, 0.0046, 0.0047, 0.0046, 0.0046, 0.0046, 0.0048, 0.0047,
        0.0048, 0.0046, 0.0048, 0.0047, 0.0046, 0.0049, 0.0046, 0.0047, 0.0046,
        0.0047, 0.0050, 0.0047, 0.0047, 0.0047, 0.0051, 0.0046, 0.0046, 0.0046,
        0.0046, 0.0049, 0.0046, 0.0047, 0.0046, 0.0046, 0.0050, 0.0046, 0.0048,
        0.0047, 0.0049, 0.0048, 0.0047, 0.0046, 0.0047, 0.0047, 0.0047, 0.0046,
        0.0048, 0.0050, 0.0048, 0.0046, 0.0050, 0.0049, 0.0048, 0.0047, 0.0050,
        0.0050, 0.0047, 0.0046, 0.0052, 0.0048, 0.0046, 0.0047, 0.0046, 0.0047,
        0.0048, 0.0046, 0.0051, 0.0048, 0.0048, 0.0049, 0.0050, 0.0048, 0.0046,
        0.0046, 0.0047, 0.0046, 0.0046, 0.0047, 0.0048, 0.0046, 0.0047, 0.0047,
        0.0047, 0.0046, 0.0048, 0.0046, 0.0046, 0.0049, 0.0047, 0.0048, 0.0049,
        0.0047, 0.0048, 0.0049, 0.0048, 0.0049, 0.0047, 0.0046, 0.0046, 0.0046,
        0.0046, 0.0047, 0.0049, 0.0048, 0.0048, 0.0047, 0.0047, 0.0049, 0.0047,
        0.0046, 0.0047, 0.0046, 0.0047, 0.0048, 0.0047, 0.0047, 0.0046, 0.0046,
        0.0048, 0.0047, 0.0048, 0.0047, 0.0047, 0.0048, 0.0047, 0.0046, 0.0052,
        0.0047, 0.0047, 0.0047, 0.0049, 0.0048, 0.0047, 0.0046, 0.0046, 0.0046,
        0.0048, 0.0046, 0.0048, 0.0047, 0.0047, 0.0046, 0.0048, 0.0046, 0.0049,
        0.0051, 0.0048, 0.0047, 0.0047, 0.0047, 0.0049, 0.0048, 0.0047, 0.0050,
        0.0047, 0.0047, 0.0049, 0.0047, 0.0046, 0.0048, 0.0048, 0.0047, 0.0046,
        0.0047, 0.0049, 0.0049, 0.0047, 0.0047, 0.0046, 0.0046, 0.0047, 0.0048,
        0.0050, 0.0047]

if __name__ == "__main__":
    get_gene_hit_counts(test_chrom, test_sig_indices, test_scores)