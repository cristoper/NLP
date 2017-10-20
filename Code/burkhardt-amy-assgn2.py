"""
This script contains all of the functions necessary to predict POS tags (includes sampling and baseline computations).
This script is executable with the following unix command line:

python burkhardt-amy-assgn2.py Data/berp-POS-training.txt Data/testset.txt > burkhardt-amy-assgn2-test-output.txt

Example code for how to call these functions to produce results is presented in the second part of the report entitled
burkhardt-amy-asgn2-report.pdf
"""
import sys
import math
import pandas as pd
import numpy as np
from random import random
import os
from collections import Counter

tags = ['CC', 'CD',
        'DT',
        'EX',
        'FW',
        'IN',
        'JJ', 'JJR', 'JJS',
        'LS',
        'MD',
        'NN', 'NNS', 'NNP', 'NNPS',
        'PDT', 'POS', 'PRP', 'PRP$',
        'RB', 'RBR', 'RBS', 'RP',
        'SYM',
        'TO',
        'UH',
        'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
        'WDT', 'WP', 'WP$', 'WRB',
        '$', '#', '"', '(', ')', ',', '.', ':'
        ]


def ngram_dict(data, ngrams = "tag_word"):
    """
    Creates dict of ngrams (key) and count (value).

    Arguments:
        data: DataFrame with 'tag' and 'word' colum
        negrams: denote type of ngram (unigram or bigram) and if want words or tags: word_word or tag_word
    Returns:
        A dict where key is either a unigram or a bigram tuple, and value is the count of the ngrams
    """
    if ngrams == "tag_tag":
        col_1 = data['tag']
        col_2 = col_1[1:col_1.shape[0]]
        ngram_count = list(zip(col_1, col_2))
        ngram_count = dict(Counter(ngram_count))
        ngram_count[('', col_1[0])] += 1

    if ngrams == "tag_word":  # not really bi-grams, just getting count of tag,word
        col_1 = data['word']
        col_2 = data['tag']
        ngram_count = list(zip(col_1, col_2))
        ngram_count = dict(Counter(ngram_count))

    if ngrams == 'tag':
        ngram_count = dict(Counter(data.tag))

    if ngrams == 'word':
        ngram_count = dict(Counter(data.word))

    return ngram_count


def fixed_vocabulary(df, min_freq=2):
    """
    Provides list of fixed vocabulary for the observation likelihood matrix

    Arguments:
        train_data: data used for training the probability matrices
        min_freq: value where if the frequency of the word in training data is less than, then the word is concered UNK

    Returns:
        vocabulary: list of vocabulary that is used as the columns of the observation matrix

    """

    unigrams = ngram_dict(df, "word")
    unknowns = {key: value for key, value in unigrams.items() if value < min_freq}
    unknowns = unknowns.fromkeys(unknowns, 'UNK')
    df['word'] = df['word'].replace(unknowns)
    vocab = ngram_dict(df, "word")
    vocabulary = list(vocab.keys())
    vocabulary.remove('')

    return vocabulary


def compute_transition_matrix (tags, bigram_counts, unigram_counts, k):
    """
    Compute probabilities for the transition matrix (len(tags)+1 x len(tags))

    Arguments:
        tags: POS tags (that may or may not appear in training data)
        bigram_counts: count of bigrams of POS tags in training data (used for numerator)
        unigram_counts: count of unigram POS tag in training data (used for denominator)
        k: laplace smoothing value

    Returns: 45 x 44 matrix of transition probabilities for all possible POS tags

    """

    transition = [] # list of transition probabilities

    # first compute the starting probabilities

    for x in tags:
            pair = ('',x) # here the period denotes the start of a sentence. Not very confident about this
            denominator = unigram_counts[''] + k*len(tags)
            try:
                 numerator = bigram_counts[pair] + k
            except:
                 numerator = k
            transition.append(math.log2(numerator / denominator))


    # then compute everything else

    for x in tags:
        for y in tags:
            pair = (x,y)
            try:
                denominator = unigram_counts[x] + k*len(tags)
            except:
                denominator = k*len(tags)
            try:
                numerator = bigram_counts[pair] + k
            except:
                numerator = k
            transition.append(math.log2(numerator / denominator))

    transition = np.array(transition)
    tran_matrix = transition.reshape(len(tags)+1, len(tags))
    return tran_matrix


def compute_observation_matrix (tags, vocabulary, bigram_counts, unigram_counts, k):
    """
    Compute probabilities for the observation matrix (tags, vocabulary)

    Arguments:
        tags: POS tags (that may or may not appear in training data)
        vocabulary: words that appear in the training set. Any words that appear less than 2 times = UNK
        bigram_counts: count of bigrams of (tag, word) (used for numerator)
        unigram_counts: count of unigram POS tag in training data (used for denominator)

    Returns: len(tags) x len(vocabulary) matrix of transition probabilities for all possible POS tags

    """

    observations = [] # list of observation likelihoods
    for x in tags:
        for y in vocabulary:
            pair = (y, x)
            try:
                denominator = unigram_counts[x] + k *len(vocabulary)
            except:
                 denominator = k*len(vocabulary)
            try:
                 numerator = bigram_counts[pair] + k
            except: numerator = k
            observations.append(math.log2(numerator / denominator))

    observations = np.array(observations)
    obs_matrix = observations.reshape(len(tags),len(vocabulary))
    return obs_matrix


def viterbi (transition, observations, events):
    """ Computes sequnce of hidden states, given observed events.
    Arguments:
        transition: transition matrix with start probabilites as first row
        observations: observation liklihood matrix, with states as rows, and vocabulary as columns
        events: sequence of observed events

    Returns:
        generator, which yields the states
    """

    n_states = transition.shape[1]
    n_events = len(events)
    v = np.zeros((n_states, n_events))
    bp = v.copy()

    # initialization step
    for s in range(n_states):
        v[s,0] = transition[0,s] + observations[s, events[0]]

    # induction step
    for t in range (1, n_events):
        for s in range(n_states):
            tmp = []
            for s_prime in range (n_states):
                prev_t = v[s_prime, t-1]
                tran_s_prime_to_s = transition[s_prime + 1, s]
                obser_s_given_t = observations[s, events[t]]
                tmp.append(prev_t + tran_s_prime_to_s + obser_s_given_t) # adding because all probabilities are in logs
            # now that all interim probabilities have been computed for given state, get max
            # and also store the index of the argmax
            v[s,t] = max(tmp) # log will be negative; so insetad of
            bp[s,t] = np.argmax(tmp) # take argmin

    # termination step
    q = np.argmax(v[:, n_events-1]) # want to get the argmax of the final time -- it will return a state index

    # back reference step
    for i in reversed(range(n_events)):
        yield q
        q = int(bp[q,i])


def get_sequence(viterbi_gen, names_events):
    """ translate viterbi generater into a sequence of state names
    """
    sequence = []
    for state in viterbi_gen:
        name = names_events[state]
        sequence.insert(0, name)

    return sequence


def get_data(path):
    df = pd.read_table("{}".format(path), '\t',
                          header=None,
                          skip_blank_lines=False,
                          keep_default_na=False,
                          names=['word_Num', 'word', 'tag'])
    return df

def get_vocabulary(df, min_freq=2):
    """
    Provides list of fixed vocabulary for the observation likelihood matrix

    Arguments:
        train_data: data used for training the probability matrices
        min_freq: value where if the frequency of the word in training data is less than, then the word is concered UNK

    Returns:
        vocabulary: list of vocabulary that is used as the columns of the observation matrix

    """

    unigrams = ngram_dict(df, "word")
    unknowns = {key: value for key, value in unigrams.items() if value < min_freq}
    unknowns = unknowns.fromkeys(unknowns, 'UNK')
    df['word'] = df['word'].replace(unknowns)
    vocab = ngram_dict(df, "word")
    vocabulary = list(vocab.keys())
    vocabulary.remove('')

    return vocabulary

def get_transition (df, k=1):
    bigram_tag_counts = ngram_dict(df, "tag_tag")
    unigram_tag_counts = ngram_dict(df, "tag")
    transitions = compute_transition_matrix(tags, bigram_tag_counts, unigram_tag_counts, k)
    return transitions


def get_observation (df, vocabulary, k=1):
    bigram_counts = ngram_dict(df, "tag_word")
    unigram_counts = ngram_dict(df, "tag")
    observations = compute_observation_matrix(tags, vocabulary, bigram_counts, unigram_counts, k)
    return observations


def read_in_print_out(df, transitions, observations, vocabulary, path, name):
    """
    Predicts the pos tags in a file that can be recognized to compare against the true labels.

    Arguments:
        df: either the training or validation dataset
        transitions: name of the transition probability matrix
        observations: name of the observation matrix
        path: file location of the predicted pos file
        df: full_train, partial train, test or something else

    returns:
        tab-delimited file of predicted scores
    """

    sentences = df['word'].tolist()

    def sent(seq, sep):
        g = []
        for el in seq:
            if el == sep:
                yield g
                g = []
            g.append(el)
        yield g
    g = []

    result = list(sent(sentences, ''))

    def get_events(new_sent, vocabulary):
        events = []
        for word in new_sent:
            try:
                events.append(vocabulary.index(word))
            except:
                events.append(vocabulary.index('UNK'))
        return events

    all_pos = []
    counter = 0
    for new_sent in result:
        if counter > 0:
            new_sent.pop(0)
        tagger = viterbi(transitions, observations, get_events(new_sent, vocabulary))
        sequence = get_sequence(tagger, tags)
        sequence.insert(len(sequence), '') #add space at the end
        all_pos.append(sequence)
        counter += 1

    flat_list = [item for sublist in all_pos for item in sublist]
    pos = pd.DataFrame({'col':flat_list})
    df['tag'] = pos

    #df.to_csv("{}burkhardt-amy-assign2-{}-output.txt".format(path,name), sep='\t', index=False, header=False)
    return df

def sampling(corpus_path):
    """
    Creates randomly sampled training and test sets from the entire training corpus

    Arguments:
        corpus_path: path where the entire training set is located

    Return: rand_test and rand_train

    """
    # probability of each sentence being saved in training file
    keep_prob = 0.8

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
    print("Wrote %d lines to %s" % (keep_count, train_path))
    print("Wrote %d lines to %s" % (test_count, test_path))


def baseline_dictionary(df):
    counts = df.groupby(['word', 'tag']).count().reset_index()
    count_max = counts.sort_values('word_Num', ascending=False).groupby(['word'], as_index=False).first()
    lookup = count_max.set_index('word')['tag'].to_dict()
    tag_counts = df.groupby('tag').count()
    top_tag = tag_counts['word_Num'].idxmax()
    lookup.update({'UNK': top_tag})  # for unknown tags, use most popular tag
    return lookup

def baseline_accuracy(df, lookup):
    """
    Assigns a predicted tag and computes accuracy
    Arguments:
        data: file that contains a column named 'word' and 'tag'
        lookup: dictionary of the tag associated with the word
    returns:
        accuracy: the accuracy of the file
    """
    df['predicted_tag'] = df['word'].map(lookup)
    df.fillna(value=lookup['UNK'], inplace=True)
    df['accuracy'] = df.apply(lambda x: 1 if x.tag == x.predicted_tag else 0, axis=1)
    print("the accuracy is:")
    print(df['accuracy'].mean())

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Missing arguments.")
        print("Usage: burkhardt-amy-assgn2.py trainingfile.txt testfile.txt > burkhardt-amy-assgn2-test-output.txt")

    else:
        training_filename = sys.argv[1]
        testing_filename = sys.argv[2]

        train = get_data(training_filename)  # either random sample or all data
        test = get_data(testing_filename)
        vocabulary = get_vocabulary(train, min_freq=2)
        transition = get_transition(train, k=.01)
        observation = get_observation(train, vocabulary, k=.01)
        pd.set_option('display.max_rows', -1)
        results = read_in_print_out(test, transition, observation, vocabulary, path ="Data/", name = 'test2')
        print(results.to_csv(sep='\t',index=False, header=False))




