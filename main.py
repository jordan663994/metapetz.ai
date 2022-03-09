import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf

import matplotlib.pyplot as plt
import os
import PIL
import time

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
class Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
          states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x
class OneStep(tf.keras.Model):
      def __init__(self, model, chars_from_ids, ids_from_chars, text, units, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.units = units
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.text = text
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

      @tf.function
      def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                              return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states
    
    

def initialize():
    path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    # Read, then decode for py2 compat.
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    # length of text is the number of characters in it
    vocab = sorted(set(text))
    ids_from_chars = tf.keras.layers.StringLookup(
        vocabulary=list(vocab), mask_token=None)
    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
    def text_from_ids(ids):
        return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)
    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    seq_length = 100
    examples_per_epoch = len(text)
    sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)
    def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text
    
    dataset = sequences.map(split_input_target)
    BATCH_SIZE = 64

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000

    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))

    dataset
    vocab_size = len(vocab)

    # The embedding dimension
    embedding_dim = 256

    # Number of RNN units
    rnn_units = 128
    model = Model(
    # Be sure the vocabulary size matches the `StringLookup` layers.
        vocab_size=len(ids_from_chars.get_vocabulary()),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units)
    model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    history = model.fit(
        dataset,
        epochs=10
    )
    model = OneStep(model, chars_from_ids, ids_from_chars, text, rnn_units)
    return [model]

def makeText2(one_step_model):
    start = time.time()
    states = None
    next_char = tf.constant(['The'])
    result = [next_char]

    for n in range(1000000):
        progress = n / 1000000
        print(f"generating... ({progress * 100}% complete)")
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)
    end = time.time()
    return tf.strings.join(result)[0].numpy().decode("utf-8")
def makeText(one_step_model, i):
    start = time.time()
    states = None
    next_char = tf.constant(['The'])
    result = [next_char]

    for n in range(200):
        progress = n / 200
        print(f"generating... ({progress * 100}% complete)")
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)
    end = time.time()
    return tf.strings.join(result)[0].numpy().decode("utf-8")
    
def grow(sequence, text, units):
    c = 1
    for i in sequence:
        text += makeText(i, c)
        c += 1
    del c
    vocab = sorted(set(text))
    ids_from_chars = tf.keras.layers.StringLookup(
        vocabulary=list(vocab), mask_token=None)
    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
    def text_from_ids(ids):
        return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)
    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    seq_length = 100
    examples_per_epoch = len(text)
    sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)
    def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text
    
    dataset = sequences.map(split_input_target)
    BATCH_SIZE = 64

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000

    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))

    dataset
    vocab_size = len(vocab)

    # The embedding dimension
    embedding_dim = 256

    # Number of RNN units
    rnn_units = units
    model = Model(
    # Be sure the vocabulary size matches the `StringLookup` layers.
        vocab_size=len(ids_from_chars.get_vocabulary()),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units)
    model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    history = model.fit(
        dataset,
        epochs=2
    )
    model = OneStep(model, chars_from_ids, ids_from_chars, text, rnn_units)
    sequence.append(model)
    return sequence
def save(sequence):
    counter = 0
    for i in range(len(sequence)):
        sequence[i].save_weights(str(i))
        text = sequence[i].text
        f = open(str(i) + ".textcache", "w")
        f.write(text)
        f.close()
        units = sequence[i].units
        f = open(str(i) + ".intcache", "w")
        f.write(str(units))
        f.close()
        counter += 1
    f = open("size.txt", "r")
    prev = int(f.read())
    f.close()
    if prev < counter:
        f = open("size.txt", "w")
        f.write(str(counter))
        f.close()
    
    
def load(latest):
    f = open("size.txt", "r")
    size = f.read()
    f.close()
    stopper = 0
    if latest:
        stopper = int(size) - 1
    seq = []
    for i in range(stopper, int(size)):
        f = open(str(i) + ".textcache")
        text = f.read()
        f.close()
        f = open(str(i) + ".intcache")
        units = int(f.read())
        f.close()
        vocab = sorted(set(text))
        ids_from_chars = tf.keras.layers.StringLookup(
            vocabulary=list(vocab), mask_token=None)
        chars_from_ids = tf.keras.layers.StringLookup(
            vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
        def text_from_ids(ids):
            return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)
        all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
        ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
        seq_length = 100
        examples_per_epoch = len(text)
        sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)
        def split_input_target(sequence):
            input_text = sequence[:-1]
            target_text = sequence[1:]
            return input_text, target_text

        dataset = sequences.map(split_input_target)
        BATCH_SIZE = 64

        # Buffer size to shuffle the dataset
        # (TF data is designed to work with possibly infinite sequences,
        # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
        # it maintains a buffer in which it shuffles elements).
        BUFFER_SIZE = 10000

        dataset = (
            dataset
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))

        dataset
        vocab_size = len(vocab)

        # The embedding dimension
        embedding_dim = 256

        # Number of RNN units
        rnn_units = units
        model = Model(
        # Be sure the vocabulary size matches the `StringLookup` layers.
            vocab_size=len(ids_from_chars.get_vocabulary()),
            embedding_dim=embedding_dim,
            rnn_units=rnn_units)
        model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
        history = model.fit(
            dataset,
            epochs=1
        )
        model = OneStep(model, chars_from_ids, ids_from_chars, text, units)
        model.load_weights(str(i))
        seq.append(model)
    return seq
running = True
c = 0
sequence = load(False)
while running:
    txtFile = input("enter the 'English Source' filename")
    f = open(txtFile, "r")
    text = f.read()
    f.close()
    t = text
    for i in range(10):
        text += " " + t
    sequence = grow(sequence, text, 2048 * 2)
    f = open(f"output{c}.txt", "w")
    f.write(makeText2(sequence[len(sequence) - 1]))
    f.close()
    save(sequence)
    if input("enter 'Y' to continue") != "Y":
        running = False