# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math
import numpy as np
class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels,DATUM_WIDTH,DATUM_HEIGHT):
    self.DATUM_WIDTH = DATUM_WIDTH
    self.DATUM_HEIGHT = DATUM_HEIGHT
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.numtrain = len(trainingLabels)
    self.numvalid = len(validationLabels)
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    self.p_y = util.Counter()
    self.p_y.incrementAll(trainingLabels, 1)
    self.p_y.divideAll(self.numtrain)

    self.c_fi_y = {}
    for label in self.legalLabels:
      self.c_fi_y[label] = util.Counter()  # this is the data-structure you should use
    for i in range(self.numtrain):
      datum = trainingData[i]
      label = trainingLabels[i]
      self.c_fi_y[label] += datum

    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20,50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels,kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"
    self.post = {}
    ans_k_acc = util.Counter()
    for k in kgrid:
      self.post.clear()
      for y in self.legalLabels:
        self.post[y] = self.c_fi_y[y].copy()
        self.post[y].incrementAll(self.post[y].keys(),k)
        self.post[y].divideAll(self.numtrain + 2 * k)
      guesses = self.classify(validationData)
      correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
      ans_k_acc[k] = correct

    ans_k_acc.divideAll(self.numvalid)
    self.ans_k = ans_k_acc.argMax()
    self.post.clear()
    for y in range(len(self.legalLabels)):
      self.post[y] = self.c_fi_y[y].copy()
      self.post[y].incrementAll(self.post[y].keys(), self.ans_k)
      self.post[y].divideAll(self.numtrain + 2 * self.ans_k)


  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses

  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    "*** YOUR CODE HERE ***"

    logJoint = util.Counter()
    for y in self.legalLabels:
      logJoint[y] += math.log(self.p_y[y])
      for a in range(self.DATUM_WIDTH):
        for b in range(self.DATUM_HEIGHT):
          if (datum[(a, b)] == 1):
            logJoint[y] += math.log(self.post[y][(a, b)])
          else:
            logJoint[y] += math.log(1-self.post[y][(a, b)])
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
       
    "*** YOUR CODE HERE ***"
    # we dont know
    util.raiseNotDefined()

    return featuresOdds
    

    
      
