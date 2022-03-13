
import numpy as np
from numpy import random
import tensorflow_datasets as tfds
import tensorflow as tf


import os
import PIL
import time

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf

import os
import PIL
import time

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
class Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units, temperature=1.0):
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
      def __init__(self, model, chars_from_ids, ids_from_chars, name, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.ids_from_chars([name])[:, None]
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
def initialize(text, vocab):
    # length of text is the number of characters in it
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
    rnn_units = 1024
    model = Model(
    # Be sure the vocabulary size matches the `StringLookup` layers.
        vocab_size=len(ids_from_chars.get_vocabulary()),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,)
    
    model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    history = model.fit(
        dataset,
        epochs=100
    )
    
    return model


def makeText(one_step_model, name):
    out2 = ""
    states = None
    for i in range(10):
        result = []
        res = []
        start = time.time()
        next_char = tf.constant(["\n \n"+name +":"+ "\n \n"+ input("You:   ")+"\n \n"])
        result.append(next_char)
        a = False
        b = random.randint(0,100)
        for i in range(b):
            progress = i / b
            print(f"generating... ({progress * 100}% complete)")
            next_char, states = one_step_model.generate_one_step(next_char, states=states)
            result.append(next_char)
            res.append(next_char)
        end = time.time()
        out = tf.strings.join(result)[0].numpy().decode("utf-8")
        out2 += "\n"
        out2 += out
        print(tf.strings.join(res)[0].numpy().decode("utf-8"))
    return out2
def chat(model, textpool, vocab, name):
    text = textpool
    # length of text is the number of characters in it
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
    
    history = model.fit(dataset, epochs=5)
    del history
    generator = OneStep(model, chars_from_ids, ids_from_chars, name)
    out = makeText(generator, name)
    print(out)
    text += "\n" + out
    return model, text
def save(model, textpool):
    model.save_weights("test3")
    f = open("textpool3.textcache", "w")
    f.write(textpool)
    f.close()
def load(text, vocab):
    units=1024
    dim=256
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
    model = Model(
    # Be sure the vocabulary size matches the `StringLookup` layers.
        vocab_size=len(ids_from_chars.get_vocabulary()),
        embedding_dim=dim,
        rnn_units=units)
    
    model.load_weights("test3")
    model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model



path_to_file = "training_data.textcache"

name = input("Enter your name")

vocab_source = open(path_to_file, 'r').read()
text = vocab_source[len(vocab_source) - 50000:]
vocab = sorted(set(vocab_source))
ini_text = text
ini_text += text
model = initialize(ini_text, vocab)
#main loop

for i in range(0, 25):
    model, text = chat(model, text, vocab, name)
    text = text[len(text) - 50000:]
    save(model, text)
