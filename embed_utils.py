import numpy as np


def generate_contexts(seq_kmer_onehots, half_window_size, window_weights = 'uniform'):

    contexts = []

    for i in range(seq_kmer_onehots.shape[0]):

        if i - half_window_size < 0:

            top_half_context = seq_kmer_onehots[0 : i, :]

        else:

            top_half_context = seq_kmer_onehots[i - half_window_size : i, :]

        if i + half_window_size + 1 > seq_kmer_onehots.shape[0]:

            bottom_half_context = seq_kmer_onehots[i + 1 : , :]

        else:

            bottom_half_context = seq_kmer_onehots[i + 1 : i + half_window_size + 1, :]

        if window_weights == 'uniform':

            th = np.sum(top_half_context, axis = 0)

            bh = np.sum(bottom_half_context, axis = 0)

            if th.shape == 0:

                contexts.append(bh)

            elif bh.shape == 0:

                contexts.append(th)

            else:

                contexts.append(th + bh)

        # Build non uniform weights here
        # Tricky part is the very beginning and very end
        #the truncated top_half and bottom_half aren't ordered

    contexts = np.array(contexts)

    context_row_sums = contexts.sum(axis = 1)

    return contexts / context_row_sums[:, np.newaxis]
