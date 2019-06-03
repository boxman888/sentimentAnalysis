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

       words.append(re.sub(r'\d+', '', bucket[0])) 
       classes.append(int(bucket[1].strip(' \n')))

    return ''.join([(row+"\n") for row in words]), np.asarray(classes)

class BagOWords:
    def __init__(self, train, test):
        self.trainData = train
        self.testData = test
        self.vocabulary = None
        self.featuresTest = None
        self.featuresTrain = None

    def __extractWords(self, sentence):
        words = re.sub("[^\w]", " ", sentence).split()
        words_cleaned =[w.lower() for w in words]
        return words_cleaned

    def __tokenize(self, sentences):
      sentences = self.__extractWords(sentences)
      words = []
      for sentence in sentences:
          w = self.__extractWords(sentence)
          words.extend(w)

      words = sorted(list(set(words)))
      return words

    def __bagofwords(self, sentence, words):
      sentence_words = self.__extractWords(sentence)
      bag = np.zeros(len(words))
      for sw in sentence_words:
        for i, word in enumerate(words):
          if word == sw:
            bag[i] += 1

      return np.asarray(bag, dtype=int)

    def __saveFile(self, FILE, vocab, features):
      with open(FILE, "w") as fp:
        fp.write(','.join(vocab) + ',classlabel\n')
        for row in features:
          r = row.astype(str)
          r = r.tolist()
          fp.write(','.join(r) + '\n')

    def __genFeatures(self, tweets, vocab, classes):
      features = []
      for i, tweet in enumerate(tweets):
        frequency = self.__bagofwords(tweet, vocab)
        frequency = np.append(frequency, [classes[i]])
        features.append(frequency)

      return features

    def process(self, train_classes, test_classes):
      vocabulary = self.__tokenize(self.trainData)
      self.vocabulary = vocabulary

      train_tweets = self.trainData.split('\n')
      train_tweets = list(filter(None, train_tweets))
      
      features_train = self.__genFeatures(train_tweets, vocabulary, train_classes)
      self.__saveFile("preprocessed_train.txt", vocabulary, features_train)
      
      test_tweets = self.testData.split('\n')
      test_tweets = list(filter(None, test_tweets))
     
      features_test = self.__genFeatures(test_tweets, vocabulary, test_classes) 
      self.__saveFile("preprocessed_test.txt", vocabulary, features_test)

      self.featuresTest = features_test
      self.featuresTrain = features_train

    #   print(len(train_classes), len(test_classes))
    #   print(len(train_tweets), len(test_tweets))
      return [(train_tweets[i], train_classes[i]) for i in range(len(train_classes))], [(test_tweets[i], test_classes[i]) for i in range(len(test_classes))]

class Classify:
    def __init__(self, traning, testing, bagObj):
      self.trainingMap = np.asarray(traning)
      self.vocab = bagObj.vocabulary
      self.testingFeatures = bagObj.featuresTest
      self.trainingFeatures = bagObj.featuresTrain
      self.testingMap = np.asarray(testing)

    def getPosDistribution(self):
      return np.sum((self.trainingMap[:, 1].astype(int))) / len(self.trainingMap)

    def probGenerator(self, classID):
      wordCount = len(self.vocab)+1
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
    
    def Predict(self, probRatio, weightOfPos, weightOfNeg, data):
      classify = []
      for feature in data:
        count = 0
        for i, w in enumerate(feature):
          if w == 1:
            count += np.log(weightOfPos[i] / weightOfNeg[i])
        if probRatio + count > 0.0:
          classify.append(1)
        else:
          classify.append(0)
       
      return classify

    def accuracy(self, predictions, data):
      count = 0
      for i in range(len(predictions)):
        if predictions[i] == int(data[i, 1]):
          count += 1
      return (count / len(predictions)) * 100
                  
    def process(self):
      pos = self.getPosDistribution()
      neg = 1 - pos
      probRatio = math.log(pos/neg)

      weightOfPos = self.probGenerator(1)
      weightOfNeg = self.probGenerator(0)
      
      predictions = self.Predict(probRatio, weightOfPos, weightOfNeg, self.trainingFeatures)

      # Print to stdout
      print("Training: trainingSet.txt")
      print("Testing:  trainingSet.txt")
      print("Accuracy: " + str(self.accuracy(predictions, self.trainingMap)) + "\n")

      # Write to output file
      fp = open("results.txt", "w")
      fp.write("Training: trainingSet.txt\n")
      fp.write("Testing:  trainingSet.txt\n")
      fp.write("Accuracy: " + str(self.accuracy(predictions, self.trainingMap)) + "\n\n")
      
      predictions = self.Predict(probRatio, weightOfPos, weightOfNeg, self.testingFeatures)

      # Print to stdout
      print("Training: trainingSet.txt")
      print("Testing:  testSet.txt")
      print("Accuracy: " + str(self.accuracy(predictions, self.testingMap)))

      # Write to output file
      fp.write("Training: trainingSet.txt\n")
      fp.write("Testing:  testSet.txt\n")
      fp.write("Accuracy: " + str(self.accuracy(predictions, self.testingMap)) + "\n")
      fp.close()

if __name__ == "__main__":
   train = LoadData("trainingSet.txt")
   test = LoadData("testSet.txt")
   
   # Could return a tuple of tweets and classes here so we do not have to do it during the 
   # processing faze. But, I am felling lazy...
   train_tweets, train_classes = train.getData()
   test_tweets, test_classes = test.getData()
   
   dictionary = BagOWords(train_tweets, test_tweets)

   # Return the processed tweets for train and test. Now a tuple with (string, bool)
   # This allows us to map a users tweet to a positive or negative feeling.
   train_tweets, test_tweets = dictionary.process(train_classes, test_classes)

   classify = Classify(train_tweets, test_tweets, dictionary)
   classify.process()
