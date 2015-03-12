# -*- coding: utf-8 -*-
"""
Preprocessing text for statistical semantics using gensim: http://radimrehurek.com/gensim/
__author__ : Michele Trevisiol @trevi
URL: http://radimrehurek.com/2014/02/word2vec-tutorial/
"""

## import modules and set up logging
# %load_ext autoreload
# %autoreload 2
from ext.TextProcessing import *
from gensim.models import word2vec
import logging
import sys
import os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
##



''' -------------------------- '''
'''   DEFINE CREATIVE OBJECT   '''
''' -------------------------- '''
class GensimCore:

  def __init__(self):
    self.load_model()


  def load_model(self,model_path=''):
    last_model_path = 'models/yahoo-datapack-20150103-1m-kw-adcopy.model'
    ## Get Last model path
    if len(model_path) == 0:
      self.model = self.get_default_model()
    ## Otherwise load the default models
    else:
      self.model = self.get_model(model_path)
    ##
  

  def get_model(self, model_path):
    return word2vec.Word2Vec.load( model_path )
    ##


  ''' Return the last model (incrementally updating all the models) '''
  def get_default_model(self):  
    ## ========================= ##
    ## Load text8 Corpus Dataset ##
    DATASET = 'corpus/text8'
    MODEL_PATH = 'models/text8.model'
    logging.info('loading dataset [%s]' %DATASET)
    if os.path.isfile(MODEL_PATH):
      model = self.get_model( MODEL_PATH )
      logging.info('\tdone [%s]' %MODEL_PATH)
      print model
    else:
      sentences = word2vec.Text8Corpus(DATASET)## 
      ## train the skip-gram model; default window=5
      model = word2vec.Word2Vec(sentences, min_count=5) # min_count of words frequency
      ## The size of the NN layers correspond to the “degrees” of freedom the training algorithm has.
      ## Bigger size values require more training data, but can lead to better (more accurate) models. 
      ## Reasonable values are in the tens to hundreds
      model = word2vec.Word2Vec(sentences, size=100)
      ## Set up Paralelization 
      model = word2vec.Word2Vec(sentences, workers=6) # default = 1 worker = no parallelization
      ## Store the model
      model.save( MODEL_PATH )
      logging.info('--- saved model (%s) based on %s'%(MODEL_PATH,DATASET) )
    ## ============================ ##
    ## Load UCL Advertising Dataset ##
    DATASET = 'corpus/ucl-open-advertising-dataset.kw.adcopy'
    MODEL_PATH = 'models/ucl-advertising.model'
    logging.info('loading dataset [%s]' %DATASET)
    if os.path.isfile(MODEL_PATH):
      model = self.get_model( MODEL_PATH )
      logging.info('\tdone [%s]' %MODEL_PATH)
    else:
      sentences = self.load_sentences_from_file(DATASET)
      model.train(sentences)
      ## Training
      model = word2vec.Word2Vec(sentences, min_count=5)
      model = word2vec.Word2Vec(sentences, size=100)
      model = word2vec.Word2Vec(sentences, workers=6)
      ## Store the model
      model.save( MODEL_PATH )
      logging.info('--- saved model (%s) based on %s'%(MODEL_PATH,DATASET) )
    ## ================================== ##
    ## Load Yahoo Datapack Corpus Dataset ##
    DATASET = 'corpus/yahoo-datapack-20150103-1m.kw.adcopy'
    MODEL_PATH = 'models/yahoo-datapack-20150103-1m-kw-adcopy.model'
    logging.info('loading dataset [%s]' %DATASET)
    if os.path.isfile(MODEL_PATH):
      model = self.get_model( MODEL_PATH )
      logging.info('\tdone [%s]' %MODEL_PATH)
    else:
      sentences = self.load_sentences_from_file(DATASET)
      model.train(sentences)
      ## Training
      model = word2vec.Word2Vec(sentences, min_count=5)
      model = word2vec.Word2Vec(sentences, size=100)
      model = word2vec.Word2Vec(sentences, workers=6)
      ## Store the model
      model.save( MODEL_PATH )
      logging.info('--- saved model (%s) based on %s'%(MODEL_PATH,DATASET) )
      ##
    return model
    ##


  ''' Given a file path load and pre-process the sentences '''
  def load_sentences_from_file(self, file_path):
    ## Loading and cleaning input file
    sentences = []
    n_words = 0
    logging.info('loading Dataset %s' %file_path)
    for line in open(file_path):
      ## Get language
      lang = get_best_language(line.strip())
      ## Clean Text
      l_words = remove_punctuation_and_number(line.strip(),lower=True)
      l_words = clean_as_keywords(l_words)
      ## Remove Stopwords
      l_words = remove_stopwords(l_words,lang)
      ## Update list
      if len(l_words) > 0:
        sentences.append(l_words)
        n_words += len(l_words)
    logging.info('loaded %d sentences with %d words'%(len(sentences),n_words) )
    return sentences
    ## 


  ''' Evaluate the model. 
      http://word2vec.googlecode.com/svn/trunk/questions-words.txt '''
  def evaluate(self, eval_file_path='ext/evaluation-questions-words.txt'):
    self.model.accuracy(eval_file_path)



gs = GensimCore()
gs.evaluate()

sys.exit()




# ... and some hours later... just as advertised...
model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
[('queen', 0.5359965)]
 
# pickle the entire model to disk, so we can load&resume training later
model.save('/tmp/text8.model')
# store the learned weights, in a format the original C tool understands
model.save_word2vec_format('/tmp/text8.model.bin', binary=True)
# or, import word weights created by the (faster) C word2vec
# this way, you can switch between the C/Python toolkits easily
model = word2vec.Word2Vec.load_word2vec_format('/tmp/vectors.bin', binary=True)
 
# "boy" is to "father" as "girl" is to ...?
model.most_similar(['girl', 'father'], ['boy'], topn=3)
more_examples = ["he his she", "big bigger bad", "going went being"]
for example in more_examples:
  a, b, x = example.split()
  predicted = model.most_similar([x, b], [a])[0][0]
  print "'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted)
 
# which word doesn't go with the others?
model.doesnt_match("breakfast cereal dinner lunch".split())
'cereal'