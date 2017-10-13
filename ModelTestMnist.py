from NeuralNetworkModel import NeuralNetworkModel
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def generateImage(file):		
	img = Image.open(file).convert('LA')
	img.save(file)

	x = cv.imread(file,0)
	data = np.asarray((255-x)/255.0)
	return data, np.reshape(data, (1,data.size))


mnist = input_data.read_data_sets("data",one_hot=True)
#trainingSet, trainingLabels = mnist.train.next_batch(mnist.train.num_examples)
trainingSet = mnist.train.images 
trainingLabels = mnist.train.labels

testingSet = mnist.test.images 
testingLabels = mnist.test.labels

myModel = NeuralNetworkModel(inputSize=784,hlSize=[500,500,500],classes=10)

#Select this if you want to train a new model
myModel.trainModel(trainingSet,trainingLabels, testingSet, testingLabels, feedForwardCycles=10, batchSize=100,debugInfo=True)
myModel.saveModel('mnistModel')

#Select this if you already trained a model which you want to reuse
#myModel.loadModel('mnistModel')

images = ['Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine']

for num in images:
	number,numberArray = generateImage('Examples/%s.png' % (num))
	prediction, highestProbability = myModel.predict(numberArray)
	plt.title('Your digit is probably a %d' % (highestProbability))
	plt.imshow(number)
	plt.show()
