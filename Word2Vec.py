import collections
import math
import os
import random
import zipfile
import numpy as np
import tensorflow as tf

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data(filename='text8.zip')
print(len(words))

vocabulary_size = 40000

def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = {}
    for word, _ in count:
        dictionary[word] = len(dictionary)
    dict_slice = lambda adict, start, end: dict((k, adict[k]) for k in list(adict.keys())[start:end])
    data = []
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dic = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dic

data, count, dictionary, reverse_dic = build_dataset(words=words)
del words
print(count[:5])
print('Sample:', data[:10], [reverse_dic[i] for i in data[:10]])

data_index = 0

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        target_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in target_to_avoid:
                target = random.randint(0, span-1)
            target_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

# batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
# for i in range(8):
#     print(batch[i], reverse_dic[batch[i]], '->', labels[i, 0],
#           reverse_dic[labels[i, 0]])

batch_size = 128
embedding_size = 300
skip_window = 8
num_skips = 4

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

graph = tf.Graph()
with graph.as_default():

    train_inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(dtype=tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform(shape=[vocabulary_size, embedding_size], minval=-1.0, maxval=1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        nec_weights = tf.Variable(tf.truncated_normal(shape=[vocabulary_size, embedding_size],
                                                      stddev=1.0/math.sqrt(embedding_size)))
        nec_biases = tf.Variable(tf.zeros([vocabulary_size]))

    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nec_weights, biases=nec_biases, labels=train_labels, inputs=embed,
                                         num_sampled=num_sampled, num_classes=vocabulary_size))

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    norm = tf.sqrt(tf.reduce_mean(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    init = tf.global_variables_initializer()

num_steps = 200001

with tf.Session(graph=graph) as session:
    init.run()
    print("Initialized")

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size=batch_size, num_skips=num_skips, skip_window=skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dic[valid_examples[i]]
                top_k = 5
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log_str = "nearest to %s: " % valid_word
                for k in range(top_k):
                    close_word = reverse_dic[nearest[k]]
                    log_str = "%s %s, " % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()