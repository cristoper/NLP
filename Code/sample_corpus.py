#!/usr/bin/env python

from random import random
import os

# probability of each sentence being saved in training file
keep_prob = 0.8

# path to input corpus
corpus_path = os.path.expanduser("~/Dropbox/NLP Readings/hw 1/berp-POS-training.txt")

# path to the training output file
train_path = "rand_training.txt"

# path to the testing output file
test_path = "rand_test.txt"

# open output files
train_f = open(train_path, 'w')
test_f = open(test_path, 'w')

keep_count, test_count = 0, 0
with open(corpus_path) as f:
    sentence = ""
    for line in f:
        sentence += line
        if line == "\n":
            # We're at the end of a sentence; write it somewhere
            if random() <= keep_prob:
                # write to train_corpus
                train_f.write(sentence)
                keep_count += 1
            else:
                # write to test_corpus
                test_f.write(sentence)
                test_count += 1
            sentence = ""

# close output files
train_f.close()
test_f.close()

# report
print("Wrote %d sentences to %s" % (keep_count, train_path))
print("Wrote %d sentences to %s" % (test_count, test_path))
