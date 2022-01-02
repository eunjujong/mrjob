%%writefile spamClassifier_test.py
from mrjob.job import MRJob
from mrjob.job import MRStep
import re, os
from ast import literal_eval
from collections import Counter
import math

class SpamClassifier_test(MRJob):
  def configure_args(self):
    super(SpamClassifier_test, self).configure_args()

    self.add_passthru_arg('--alpha', default=1.0, type=int) # Laplace smoothing factor to avoid overfitting 

  def mapper_init(self):
    self.train_len = 0
    self.test_len = 0
    self.s_counts = 0
    self.h_counts = 0
    self.s_dict = Counter()
    self.h_dict = Counter()

    path = '/content/spam_train.txt' #os.path.abspath("spam_train.txt")
    with open(path, 'r') as f:
      for line in f:
        _, vals = line.split(None, 1)
        vals = literal_eval(vals)
        s_c, h_c, s_d, h_d = vals
        self.s_counts += s_c
        self.h_counts += h_c
        self.s_dict += s_d
        self.h_dict += h_d

  def mapper(self, _, lines):
    id, label, texts = lines.split(',', 2)
    word_list = re.findall(r"[\w']+", texts.lower())

    if id == 'train':
      self.train_len += 1
    else:
      self.test_len += 1

    p_s = 0
    p_h = 0
    p_c_s = 0
    p_c_h = 0
    a = self.options.alpha
    for word in word_list:
      if word in self.s_dict.keys():
        p_s += (math.log(self.s_dict[word]+a)-math.log(len(self.s_dict)+a*(len(self.s_dict) + len(self.h_dict))+1))
      elif word in self.h_dict.keys():
        p_h += (math.log(self.h_dict[word]+a)-math.log(len(self.h_dict)+a*(len(self.s_dict) + len(self.h_dict))+1))

    p_c_s = math.log(self.s_counts/(self.s_counts+self.h_counts)) + p_s
    p_c_h = math.log(self.h_counts/(self.s_counts+self.h_counts)) + p_h

    yield id, (label, p_c_s, p_c_h, self.train_len, self.test_len)

  def reducer_init(self):
    self.train_c = 0
    self.test_c = 0
    self.train_len = 0
    self.test_len = 0

  def reducer(self, id, values):
    for label, p_c_s, p_c_h, train_len, test_len in values:
      pred = 'ham'
      if 'train' in id:
        temp = train_len
        if temp > self.train_len:
          self.train_len = temp
        elif temp < self.train_len:
          self.train_len += 1
        if p_c_s > p_c_h:
          pred = 'spam'
          #yield 'Training Data', 'Initial Class: %s | Prediction: %s' % (label, pred)
        #else:
          #yield 'Training Data', 'Initial Class: %s | Prediction: %s' % (label, pred)
        if pred == label:
          self.train_c += 1
        #yield None, 'Training Accuracy: %.3f%%' % (self.train_c/train_len*100)
        yield id, (self.train_c, self.train_len)
      elif 'test' in id:
        temp = test_len
        if temp > self.test_len:
          self.test_len = temp
        elif temp < self.test_len:
          self.test_len += 1
        if p_c_s > p_c_h:
          pred = 'spam'
          #yield 'Testing Data', 'Initial Class: %s | Prediction: %s' % (label, pred)
        #else:
          #yield 'Testing Data', 'Initial Class: %s | Prediction: %s' % (label, pred)
        if pred == label:
          self.test_c += 1
        #yield None, 'Testing Accuracy: %.3f%%' % (self.test_c/test_len*100)
        yield id, (self.test_c, self.test_len)

  def mapper2(self, id, values):
    yield id, (values[0], values[1])

  def reducer2(self, id, values):
    final_count = 0
    final_len = 0
    for count, len in values:
      if 'train' in id:
        final_count += count
        final_len += len
      elif 'test' in id:
        final_count += count
        final_len += len

    if 'train' in id:
      yield None, 'Training Accuracy: %.3f%%' % (count/len*100)
    else:
      yield None, 'Testing Accuracy: %.3f%%' % (count/len*100)
  
  def steps(self):
    return [
      MRStep(mapper_init=self.mapper_init,
             mapper=self.mapper,
             reducer_init=self.reducer_init,
             reducer=self.reducer),
      MRStep(mapper=self.mapper2,
             reducer=self.reducer2)
    ]

if __name__ == '__main__':
  SpamClassifier_test.run()