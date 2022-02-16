# dataClassifier.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# This file contains feature extraction methods and harness 
# code for data classification
import time

import naiveBayes
import perceptron
import knn
import samples
import sys
import util
import numpy

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70
digit_train = 5000
digit_validation = 1000
digit_test = 1000
face_train = 451
face_validation = 150
face_test =  150


def basicFeatureExtractorDigit(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is white (0) or gray/black (1)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def basicFeatureExtractorFace(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is an edge (1) or no edge (0)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(FACE_DATUM_WIDTH):
    for y in range(FACE_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def enhancedFeatureExtractorDigit(datum):
  """
  Your feature extraction playground.
  
  You should return a util.Counter() of features
  for this datum (datum is of type samples.Datum).
  
  ## DESCRIBE YOUR ENHANCED FEATURES HERE...
  
  ##
  """
  a = datum.getPixels()

  features = util.Counter()
  num = 20
  for x in range(FACE_DATUM_WIDTH/20):
    for y in range(FACE_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x, y)] = 1
      else:
        features[(x, y)] = 0
  return features
  
  return features


def contestFeatureExtractorDigit(datum):
  """
  Specify features to use for the minicontest
  """
  features =  basicFeatureExtractorDigit(datum)
  return features

def enhancedFeatureExtractorFace(datum):
  """
  Your feature extraction playground for faces.
  It is your choice to modify this.
  """
  features =  basicFeatureExtractorFace(datum)
  return features

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
  """
  This function is called after learning.
  Include any code that you want here to help you analyze your results.
  
  Use the printImage(<list of pixels>) function to visualize features.
  
  An example of use has been given to you.
  
  - classifier is the trained classifier
  - guesses is the list of labels predicted by your classifier on the test set
  - testLabels is the list of true labels
  - testData is the list of training datapoints (as util.Counter of features)
  - rawTestData is the list of training datapoints (as samples.Datum)
  - printImage is a method to visualize the features 
  (see its use in the odds ratio part in runClassifier method)
  
  This code won't be evaluated. It is for your own optional use
  (and you can modify the signature if you want).
  """
  
  # Put any code here...
  # Example of use:
  for i in range(len(guesses)):
      prediction = guesses[i]
      truth = testLabels[i]
      if (prediction != truth):
          print ("===================================")
          print ("Mistake on example %d" % i )
          print ("Predicted %d; truth is %d" % (prediction, truth))
          print ("Image: ")
          print (rawTestData[i])
          break


## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
      self.width = width
      self.height = height

    def printImage(self, pixels):
      """
      Prints a Datum object that contains all pixels in the 
      provided list of pixels.  This will serve as a helper function
      to the analysis function you write.
      
      Pixels should take the form 
      [(2,2), (2, 3), ...] 
      where each tuple represents a pixel.
      """
      image = samples.Datum(None,self.width,self.height)
      for pix in pixels:
        try:
            # This is so that new features that you could define which 
            # which are not of the form of (x,y) will not break
            # this image printer...
            x,y = pix
            image.pixels[x][y] = 2
        except:
            print ("new features:", pix)
            continue
      print (image)

def default(str):
  return str + ' [Default: %default]'

def readCommand( argv ,p,c_str):
  "Processes the command used to run from the command line."
  from optparse import OptionParser  
  parser = OptionParser(USAGE_STRING)

  parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=[ 'nb', 'naiveBayes', 'perceptron', 'knn'], default='knn')
  parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces'], default='digits')
  parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
  parser.add_option('-p', '--training_percentage', help=default('The percentage size of the training set'), default=1.0, type="float")
  parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
  parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
  parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=2.0)
  parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
  parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=digit_test, type="int")

  options, otherjunk = parser.parse_args(argv)
  if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
  args = {}
  options.training_percentage = p
  options.classifier = c_str
  # Set up variables according to the command line input.
  print ("Doing classification")
  print ("--------------------")
  print ("data:\t\t" + options.data)
  print ("classifier:\t\t" + options.classifier)

  DATUM_WIDTH = DIGIT_DATUM_WIDTH
  DATUM_HEIGHT = DIGIT_DATUM_HEIGHT
  if(options.data=="digits"):
    print("training set size:\t" + str(int(options.training_percentage *digit_train))+ ','+str(int(p*100))+'%')
    options.training = int(options.training_percentage *digit_train)
    printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
    if (options.features):
      featureFunction = enhancedFeatureExtractorDigit
    else:
      featureFunction = basicFeatureExtractorDigit

    options.test = digit_test
  elif(options.data=="faces"):
    print("training set size:\t" + str(int(options.training_percentage * face_train)))
    options.training = int(options.training_percentage * face_train)
    printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
    if (options.features):
      featureFunction = enhancedFeatureExtractorFace
    else:
      featureFunction = basicFeatureExtractorFace
    DATUM_WIDTH = FACE_DATUM_WIDTH
    DATUM_HEIGHT = FACE_DATUM_HEIGHT
    options.test = face_test
  else:
    print ("Unknown dataset", options.data)
    print (USAGE_STRING)
    sys.exit(2)
    
  if(options.data=="digits"):
    legalLabels = range(10)
  else:
    legalLabels = range(2)
    
  if options.training_percentage <= 0 or options.training_percentage>1 :
    print ("Training set size should be a positive float (0,1] (you provided: %f)" % options.training_percentage)
    print (USAGE_STRING)
    sys.exit(2)
    
  if options.smoothing <= 0:
    print ("Please provide a positive number for smoothing (you provided: %f)" % options.smoothing)
    print (USAGE_STRING)
    sys.exit(2)
    


  if(options.classifier == "naiveBayes" or options.classifier == "nb"):
    classifier = naiveBayes.NaiveBayesClassifier(legalLabels,DATUM_WIDTH,DATUM_HEIGHT)
    classifier.setSmoothing(options.smoothing)
  elif(options.classifier == "perceptron"):
    classifier = perceptron.PerceptronClassifier(legalLabels,options.iterations)
  elif(options.classifier == "knn"):
    classifier = knn.KnnClassifier(legalLabels,DATUM_WIDTH,DATUM_HEIGHT)
  else:
    print ("Unknown classifier:", options.classifier)
    print (USAGE_STRING)
    
    sys.exit(2)

  args['classifier'] = classifier
  args['featureFunction'] = featureFunction
  args['printImage'] = printImage


  
  return args, options

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -p 1.0 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 100% of Digits training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """

# Main harness code

def runClassifier(args, options):

  featureFunction = args['featureFunction']
  classifier = args['classifier']
  printImage = args['printImage']

  # Load data  
  numTraining = options.training
  numTest = options.test

  if(options.data=="faces"):
    rawTrainingData = samples.loadDataFile("./data/facedata/facedatatrain", numTraining,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("./data/facedata/facedatatrainlabels", numTraining)
    rawValidationData = samples.loadDataFile("./data/facedata/facedatatrain", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("./data/facedata/facedatatrainlabels", numTest)
    rawTestData = samples.loadDataFile("./data/facedata/facedatatest", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("./data/facedata/facedatatestlabels", numTest)
  else:
    rawTrainingData = samples.loadDataFile("./data/digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("./data/digitdata/traininglabels", numTraining)
    rawValidationData = samples.loadDataFile("./data/digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("./data/digitdata/validationlabels", numTest)
    rawTestData = samples.loadDataFile("./data/digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("./data/digitdata/testlabels", numTest)
    
  
  # Extract features
  # print ("Extracting features...")
  trainingData = list(map(featureFunction, rawTrainingData))
  validationData =  list(map(featureFunction, rawValidationData))
  testData =  list(map(featureFunction, rawTestData))

  # Conduct training and testing
  # print ("Training...")
  start = time.time()
  classifier.train(trainingData, trainingLabels, validationData, validationLabels)
  end = time.time()
  # print ("Validating...")
  # guesses = classifier.classify(validationData)
  # correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
  # print (str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels)))
  # print ("Testing...")
  guesses = classifier.classify(testData)
  correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
  print (str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels)))
  print('training time : %.1f s'% (end-start) )
  return end-start, 100.0 * correct / len(testLabels)

if __name__ == '__main__':
  # Read input
  time_list = dict()
  acc_list =  dict()
  c_str_s = ['naiveBayes', 'perceptron', 'knn']
  ps = []
  for c_str in c_str_s:
    time_list[c_str] =[]
    acc_list[c_str] =[]
    for i in range(10):
      p = (i+1)*0.1
      ps.append(p)
      args, options = readCommand( sys.argv[1:],p,c_str)
      # Run classifier
      duration,acc = runClassifier(args, options)
      time_list[c_str].append(duration)
      acc_list[c_str].append(acc)

  import matplotlib.pyplot as plt
  bar_width = 0.3

  index = numpy.arange(10)

  index1 = index + bar_width
  index2 = index + bar_width*2
  name = []
  for i in range(10):
    name.append(str((i + 1) * 10) + '%')

  plt.figure()
  plt.bar(index, height=time_list['naiveBayes'], width=bar_width, color='b', label='naiveBayes')
  plt.bar(index1, height=time_list['perceptron'], width=bar_width, color='g', label='perceptron',tick_label = name)
  plt.bar(index2, height=time_list['knn'], width=bar_width, color='r', label='knn')

  plt.legend()
  plt.ylabel('train time :s')
  plt.title('train time of different number trainning data')


  plt.figure()
  plt.bar(index, height=acc_list['naiveBayes'], width=bar_width, color='b', label='naiveBayes')
  plt.bar(index1, height=acc_list['perceptron'], width=bar_width, color='g', label='perceptron',tick_label = name)
  plt.bar(index2, height=acc_list['knn'], width=bar_width, color='r', label='knn')

  plt.legend()
  plt.ylabel('accurracy :%')
  plt.title('accurracy of different number trainning data')

  plt.show()

