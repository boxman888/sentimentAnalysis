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
    return ''.join([(row+"\n") for row in words]), np.asarray(classes)

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
      sentences = self._extractWords(sentences)
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

      return np.asarray(bag,dtype=int)

    def process(self, FILE):
      vocabulary = self._tokenize(self.data)
      
      tweets = self.data.split('\n')
      
      features = []
      for tweet in tweets:
        frequency = self._bagofwords(tweet, vocabulary)
        features.append(frequency)

      with open(FILE, "w") as fp:
        fp.write(','.join(vocabulary)+',classlabel\n')
        for row in features:
          r = row.astype(str)
          r = r.tolist()
          fp.write(','.join(r)+'\n')

if __name__ == "__main__":
   train = LoadData("trainingSet.txt")
   test = LoadData("testSet.txt")

   train_book, train_classes = train.getData()
   test_book, test_classes = test.getData()
   
   train_dictionary = BagOWords(train_book)
   test_dictionary = BagOWords(test_book)

   train_dictionary.process("preprocessed_train.txt")
   test_dictionary.process("preprocessed_test.txt")


   
