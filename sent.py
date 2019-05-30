import numpy as np
import re
import sys

from string import punctuation

np.set_printoptions(threshold=sys.maxsize)

class LoadData:
  def __init__(self, FILE):
    self.filename = FILE
  
  def strip_punctuation(self, s):  
    return ''.join(c for c in s if c not in punctuation)

  def getData(self):
    words = []
    classes = []
    with open(self.filename) as fp:
      for line in fp:
       sentance = self.strip_punctuation(line)
       bucket = sentance.split('\t')

       words.append(bucket[0])
       
       classes.append(int(bucket[1].strip(' \n')))
    return ''.join([(row) for row in words]), np.asarray(classes)

class BagOWords:
    def __init__(self, data):
        self.data = data
        self.vocabulary = None
        self.frequency = None

    def _extractWords(self, sentence):
        words = re.sub("[^\w]", " ", sentence).split()
        words_cleaned =[w.lower() for w in words]
        return words_cleaned

    def _tokenize(self, sentences):
      words = []
      for sentence in sentences:
          w = self._extractWords(sentence)
          words.extend(w)

      words = sorted(list(set(words)))
      return words

    def _bagofwords(self, sentence, words):
      sentence_words = self._extractWords(sentence)
      bag = np.zeros(len(words))
      for sw in sentence_words:
        for i, word in enumerate(words):
          if word == sw:
            bag[i] += 1

      return np.array(bag)

    def process(self):
      vocabulary = self._tokenize(self.data)
      frequency = self._bagofwords(self.data, vocabulary)
      
      self.vocabulary = np.asarray(vocabulary)
      self.frequency = np.asarray(frequency)

      return self.vocabulary, self.frequency

if __name__ == "__main__":
   train = LoadData("trainingSet.txt")
   book, classes = train.getData()
   bag = BagOWords(book)
   vocabulary, frequency = bag.process()
   
