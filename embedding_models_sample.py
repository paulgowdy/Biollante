import tensorflow as tf
from biollante_utils import *
from embed_utils import *

#class simple_kmer_embedding:

sess = tf.InteractiveSession()

def kmer_embedding(input_kmer_vectors, embedding_dims, k):

    x = tf.layers.dense(input_kmer_vectors, embedding_dims)

    logits = tf.layers.dense(x, 4 ** k)

    outputs = tf.nn.softmax(logits)

    return outputs

k = 2
embedding_dims = 8

inputs = tf.placeholder(tf.float32, shape=[None, 4 ** k])
target_contexts = tf.placeholder(tf.float32, shape=[None, 4 ** k])

context_outputs = kmer_embedding(inputs, embedding_dims, k)

loss = tf.reduce_mean(tf.square(tf.subtract(context_outputs, target_contexts)))

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess.run(tf.global_variables_initializer())

z = seq_to_kmer_vector('AAATTTCCCGGG', k, 1)
b = generate_contexts(z, 2)

for i in range(20):

    print(i)
    train_step.run(feed_dict = {inputs : z, target_contexts : b})
