import numpy as np
from string import punctuation

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

       bucket[0] = bucket[0].rstrip() 
       words.append(list(bucket[0].split(" ")))
       
       classes.append(int(bucket[1].strip(' \n')))
       
    return np.asarray(words), np.asarray(classes)

if __name__ == "__main__":
   train = LoadData("trainingSet.txt")
   book, classes = train.getData()
   #print(book)
   print(classes)
