import numpy as np
from itertools import product

def kmers(seq, k, stride):

    kmers = []

    for i in range(0, len(seq) - k + 1, stride):

        kmers.append(seq[i:i + k])

    return kmers

def generate_all_kmers(k):

    return [''.join(x) for x in list(product('ATCG', repeat = k))]

def seq_to_kmer_vector(seq, k, stride):

    # Returns a list of one-hot encoded vectors (as a matrix)
    # for the input seq
    # 1's corresponding to the kmer ID at a given position

    all_kmers = generate_all_kmers(k)

    seq_kmers = kmers(seq, k, stride)

    seq_kmer_ids = []

    for km in seq_kmers:

        seq_kmer_ids.append(all_kmers.index(km))

    seq_kmer_ids = np.array(seq_kmer_ids)

    seq_kmer_vector = np.zeros((seq_kmer_ids.shape[0], len(all_kmers)))

    seq_kmer_vector[np.arange(seq_kmer_ids.shape[0]), seq_kmer_ids] = 1.0

    return seq_kmer_vector
