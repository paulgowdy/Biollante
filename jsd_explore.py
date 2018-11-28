import dit
from freqgen import *
from dit.divergences import jensen_shannon_divergence

import random
import numpy as np
import matplotlib.pyplot as plt

def create_random_seq(len_seq):

    alphabet = ['A', 'C', 'T', 'G']

    seq = ''

    for i in range(len_seq):

        seq += random.choice(alphabet)

    return seq

def vector(seq, k):
    output = k_mer_frequencies(seq, [x for x in k if x != "codons"], include_missing=True, vector=True)
    if "codons" in k:
        output = np.concatenate((output, [x[1] for x in sorted(codon_frequencies(seq, genetic_code).items(), key=lambda x: x[0])]))
    return output

sl = 200
k = 2

eds = []
jsds = []

for i in range(1000):

    rand_seq_1 = create_random_seq(sl)
    rand_seq_2 = create_random_seq(sl)

    #kmer_freqs_1 = k_mer_frequencies(rand_seq_1, k, include_missing = True)
    #kmer_freqs_2 = k_mer_frequencies(rand_seq_2, k, include_missing = True)

    #print(kmer_freqs_1)
    #print(kmer_freqs_2)

    vector_1 = vector(rand_seq_1, [k])
    vector_2 = vector(rand_seq_2, [k])

    eds.append(np.linalg.norm(vector_1 - vector_2))
    jsds.append(jensen_shannon_divergence([dit.ScalarDistribution(vector_1), dit.ScalarDistribution(vector_2)]))

plt.figure()
plt.scatter(eds, jsds, edgecolor='black', linewidth='1', alpha = 0.5, facecolor = 'green')
plt.xlabel('Euclidean Distance')
plt.ylabel('Jensen-Shannon Divergence')
plt.title('JSD vs. ED, k = ' + str(k) + ', seq_len = ' + str(sl))
plt.show()
