import numpy as np
import re
import math
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
        self.features = None

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

      return np.asarray(bag, dtype=int)

    def process(self, FILE, classes):
      vocabulary = self._tokenize(self.data)
      self.vocabulary = vocabulary

      tweets = self.data.split('\n')
      tweets = list(filter(None, tweets))

      features = []
      for i, tweet in enumerate(tweets):
        frequency = self._bagofwords(tweet, vocabulary)
        frequency = np.append(frequency, [classes[i]])
        features.append(frequency)

      with open(FILE, "w") as fp:
        fp.write(','.join(vocabulary)+',classlabel\n')
        for row in features:
          r = row.astype(str)
          r = r.tolist()
          fp.write(','.join(r) + '\n')

      self.features = features
      return [(tweets[i], classes[i]) for i in range(len(classes))]

class Classify:
    def __init__(self, traning, testing, trainingObj):
      self.trainingMap = np.asarray(traning)
      self.trainingVocab = trainingObj.vocabulary
      self.trainingFeatures = trainingObj.features
      self.testingMap = np.asarray(testing)

    def getPosDistribution(self):
      return np.sum((self.trainingMap[:, 1].astype(int))) / len(self.trainingMap)

    def probGenerator(self, classID):
      wordCount = len(self.trainingVocab)+1
      state = np.zeros(wordCount)
      found = 0
      for feature in self.trainingFeatures:
        if classID == feature[-1]:
          found += 1
          for i in range(wordCount):
            if feature[i] == 1:
              state[i] += 1

      for i in range(wordCount):
        if state[i] == 0:
          state[i] = 1.0 / wordCount
        else:
          state[i] = state[i] / found
      return state
    
    def Predict(self, probRatio, weightOfPos, weightOfNeg):
      classify = []
      for feature in self.trainingFeatures:
        count = 0
        for i, w in enumerate(feature):
          if w == 1:
            count += np.log(weightOfPos[i] / weightOfNeg[i])
        if probRatio + count > 0.0:
          classify.append(1)
        else:
          classify.append(0)
       
      return classify

    def accuracy(self, predictions):
      count = 0
      for i in range(len(predictions)):
        if predictions[i] == int(self.trainingMap[i, 1]):
          count += 1
      return (count / len(predictions)) * 100
                  
    def process(self):
      pos = self.getPosDistribution()
      neg = 1 - pos
      probRatio = math.log(pos/neg)

      weightOfPos = self.probGenerator(1)
      weightOfNeg = self.probGenerator(0)
      
      predictions = self.Predict(probRatio, weightOfPos, weightOfNeg)
      
      print(self.accuracy(predictions))
      #predict = probRatio + np.sum(np.log(weightOfPos / weightOfNeg))

if __name__ == "__main__":
   train = LoadData("trainingSet.txt")
   test = LoadData("testSet.txt")
   
   # Could return a tuple of tweets and classes here so we do not have to do it during the 
   # processing faze. But, I am felling lazy...
   train_tweets, train_classes = train.getData()
   test_tweets, test_classes = test.getData()
   
   train_dictionary = BagOWords(train_tweets)
   test_dictionary = BagOWords(test_tweets)

    
   # Return the processed tweets for train and test. Now a tuple with (string, bool)
   # This allows us to map a users tweet to a positive or negative feeling.
   train_tweets = train_dictionary.process("preprocessed_train.txt", train_classes)
   test_tweets = test_dictionary.process("preprocessed_test.txt", test_classes)
   
   classify = Classify(train_tweets, test_tweets, train_dictionary)
   classify.process()
