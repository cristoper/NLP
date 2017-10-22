from collections import Counter, defaultdict
import sys
import math


class NBClassifier:
    """
    Implements a naive Bayes classifier.

    Initialize it with a list of categories:
        >>> classifier = NBCLassifier(["class1", "class2"])

    Then train it with a list of documents. Documents are 2-tuples of the form:
    [("Text of document", "class1"), ...]
        >>> classifier.train(list_of_docs)

    Then have it predict the category of a string:
        >>> classifier.predict(some_string)
    """
    def __init__(self, categories):
        """
        Args:
            categories: list of categories (strings)
        """
        self.categories = categories
        self.Ndoc = 0
        self.catdict = defaultdict(list)  # mapping from category to words in that category
        self.catcount = defaultdict(int)  # mapping from category to num docs in that category
        self.V = set()  # the vocabulary of words

        self.logprior = {}
        self.logliklihood = {}

    def train(self, documents):
        """
        Args:
            documents: a list of tuples of strings: [(text, category), ..]
        """
        self.Ndoc += len(documents)
        for doc in documents:
            text, cat = doc
            new_words = text.split()

            # update bag of words
            self.catdict[cat].extend(new_words)

            # update count of docs in cat
            self.catcount[cat] += 1

            # update vocabulary
            self.V.update(new_words)

        # update probability of word given cat
        for cat in self.categories:
            # update logprior
            Nc = self.catcount[cat]
            if Nc == 0:
                self.logprior[cat] = sys.float_info.min_exp
            else:
                self.logprior[cat] = math.log(Nc/self.Ndoc)
            self.logprior[cat] = math.log(0.5)

            # get and count the bag of words for all documents in cat
            word_counts = Counter(self.catdict[cat])
            totals = sum(word_counts.values())

            # calculate log liklihoods for each word given cat
            # (Uses +1 smoothing)
            print(totals+len(self.V))
            for word in self.V:
                count = word_counts[word] + 1
                self.logliklihood[(word, cat)] = math.log(count/(totals+len(self.V)))

    def predict(self, text):
        """
        Args:
            text: string to classify

        returns the most likely category.
        """
        sums = []
        for cat in self.categories:
            sums.append(self.logprior[cat])
            for word in text.split():
                if word in self.V:
                    #print(word, cat)
                    #print(logliklihood.get((word, cat), sys.float_info.min_exp))
                    sums[-1] += self.logliklihood.get((word, cat), 0)
        return self.categories[sums.index(max(sums))]
