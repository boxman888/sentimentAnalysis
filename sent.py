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

    def process(self, FILE, classes):
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
      

      return [(tweets[i], classes[i]) for i in range(len(classes))]

if __name__ == "__main__":
   train = LoadData("trainingSet.txt")
   test = LoadData("testSet.txt")

   train_tweets, train_classes = train.getData()
   test_tweets, test_classes = test.getData()
   
   train_dictionary = BagOWords(train_tweets)
   test_dictionary = BagOWords(test_tweets)
   
   # Return the processed tweets for train and test. Now a tuple with (string, bool)
   # This allows us to map a users tweet to a positive or negative feeling.
   train_tweets = train_dictionary.process("preprocessed_train.txt", train_classes)
   test_tweets = test_dictionary.process("preprocessed_test.txt", test_classes)

   
