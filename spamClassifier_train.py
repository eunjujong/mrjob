from mrjob.job import MRJob
from mrjob.job import MRStep
import re
from collections import Counter, defaultdict
import math

class SpamClassifier_train(MRJob):
  def mapper(self, _, lines):
    id, label, texts = lines.split(',', 2)
    word_list = re.findall(r"[\w']+", texts.lower())

    for i in range(len(word_list)):
      yield ('id_'+str(id), word_list, label, word_list[i]), 1

  def mapper_total_word_counts(self, words_texts, counts):
    yield words_texts[0], (words_texts[1], words_texts[2], words_texts[3], counts)

  def mapper_total(self, docid, words_counts):
    yield docid, (words_counts[0], words_counts[1], words_counts[2], words_counts[3])

  def reducer_word_freq(self, words_texts, counts):  # word counts in each email
    yield (words_texts[0], words_texts[1], words_texts[2], words_texts[3]), sum(counts)

  def reducer_total_word_counts(self, docid, words_counts): # spam and ham counts
    s_dict = defaultdict(int)
    h_dict = defaultdict(int)
    for word_list, label, word, counts in words_counts:
      if label == 'spam':
        s_dict[word] += counts
      elif label == 'ham':
        h_dict[word] += counts

    yield docid, (word_list, label, s_dict, h_dict)

  def reducer_init(self):
    self.s_counts = 0
    self.h_counts = 0
    self.s_dict = Counter()
    self.h_dict = Counter()

  def reducer_total(self, docid, word_counts):  # aggregate counts for spam, ham, and vocab
    for doc, label, s, h in word_counts:
      if len(s) > 0:
        self.s_dict += Counter(s)
        self.s_counts += 1
      elif len(h) > 0:
        self.h_dict += Counter(h)
        self.h_counts += 1

  def reducer_final(self):
    yield None, (self.s_counts, self.h_counts, self.s_dict, self.h_dict)

  def steps(self):
    return [
      MRStep(mapper=self.mapper,
             reducer=self.reducer_word_freq),
      MRStep(mapper=self.mapper_total_word_counts,
             reducer=self.reducer_total_word_counts),
      MRStep(mapper=self.mapper_total,
             reducer_init=self.reducer_init,
             reducer=self.reducer_total,
             reducer_final=self.reducer_final)
    ]

if __name__ == '__main__':
  SpamClassifier_train.run()