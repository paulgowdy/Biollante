from generate import generate
from freqgen import *

from Bio import SeqIO

def create_target_dict(seq, ks):

    target = {}

    for k in ks:

        target[k] = k_mer_frequencies([seq], k, include_missing = True)

    # Deal with codon case...

    return target

seqA = SeqIO.read("sample_fasta_A.fasta", "fasta").seq
seqB = SeqIO.read("sample_fasta_B.fasta", "fasta").seq

#b_kmer_freqs = k_mer_frequencies(str(seqB), 2, include_missing = True)
b_kmer_freqs = k_mer_frequencies([seqB], 2, include_missing = True)

print(b_kmer_freqs)

target = {2: b_kmer_freqs}

b_AA = translate(seqB)
a_AA = translate(seqA)

#print(b_AA)

optimized_A = generate(target, a_AA, verbose = True)

opt_a_kmer_freqs = k_mer_frequencies([optimized_A], 2, include_missing = True)

print(opt_a_kmer_freqs)
print(a_AA == translate(optimized_A))

#print(len(b_3mers))
#print(b_3mers)
