# knn.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Mira implementation
import util
PRINT = True

class KnnClassifier:
  """
  Mira classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, DATUM_WIDTH,DATUM_HEIGHT):
    self.legalLabels = legalLabels
    self.type = "knn"
    self.DATUM_WIDTH = DATUM_WIDTH
    self.DATUM_HEIGHT = DATUM_HEIGHT


  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    "Outside shell to call your method. Do not modify this method."

    self.numtrain = len(trainingLabels)
    self.numvalid = len(validationLabels)
    self.p_y = util.Counter()
    self.p_y.incrementAll(trainingLabels, 1)

    self.mean_fi_y = {}
    for label in self.legalLabels:
      self.mean_fi_y[label] = util.Counter()  # this is the data-structure you should use
    for i in range(self.numtrain):
      datum = trainingData[i]
      label = trainingLabels[i]
      self.mean_fi_y[label] += datum
    for label in self.legalLabels:
      self.mean_fi_y[label].divideAll(self.p_y[label])

  def classify(self, testData ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    guesses = []
    for datum in testData:
      errors = self.calculateDistance(datum)
      guesses.append(errors.argMax())
    return guesses

  def calculateDistance(self, datum):
    """

    """
    "*** YOUR CODE HERE ***"

    Distance = util.Counter()
    for y in self.legalLabels:
      Distance[y] += self.mean_fi_y[y]*datum

    return Distance

  def findHighOddsFeatures(self, label1, label2):
    """
    Returns a list of the 100 features with the greatest difference in feature values
                     w_label1 - w_label2

    """
    featuresOdds = []

    "*** YOUR CODE HERE ***"

    return featuresOdds

