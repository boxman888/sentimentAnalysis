import numpy as np
import re
import sys

from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer

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

       #bucket[0] = bucket[0].rstrip() 
       #words.append(bucket[0].split(" "))
       words.append(bucket[0])
       
       classes.append(int(bucket[1].strip(' \n')))
    return [''.join([(row) for row in words])], np.asarray(classes)

class BagOWords:
    def __init__(self, data):
        self.data = data
        self.vocabulary = None
        self.frequency = None

    def _extractWords(self, sentence):
        ignore_words =['a']
        words = re.sub("[^\w]", " ", sentence).split()
        words_cleaned =[w.lower() for w in words if w not in ignore_words]
        return words_cleaned

    def _tokenize(self, sentences):
      words = []
      for sentence in sentences:
          w = self._extractWords(sentence)
          words.extend(w)

      words = sorted(list(set(words)))
      return words

    def _bagowords(self, sentence, words):
      sentence_words = self._extractWords(sentence)
      bag = np.zeros(len(words))
      for sw in self.sentence_words:
        for i, word in enumerate(words):
          if word == sw:
            bag[i] += 1

      return np.array(bag)

    def process(self):
      vocabulary = self._tokenize(self.data)
      #bagofwords(self.data, vocabulary)
      vectorizer = CountVectorizer(analyzer = "word", tokenizer = None,
              preprocessor = None, stop_words = None, max_features = 10000)
      train = vectorizer.fit_transform(self.data)
      
      self.vocabulary = np.asarray(vocabulary)
      self.frequency = np.asarray(vectorizer.transform(self.data).toarray())

      return self.vocabulary, self.frequency

if __name__ == "__main__":
   train = LoadData("trainingSet.txt")
   book, classes = train.getData()
   bag = BagOWords(book)
   vocabulary, frequency = bag.process()
   
