import tensorflow as tf
import numpy as np

class NeuralNetworkModel:
	def __init__(self,inputSize=100,hlSize=[100,100,100],classes=10):
		self.inputData = tf.placeholder('float',[None,inputSize])
		self.inputLabel = tf.placeholder('float')

		self.sess = tf.Session()

		self.model = self.defineModel(self.inputData, inputSize, hlSize, classes)

	def defineModel(self,data,inputSize,hlSize,classes):	
		if len(hlSize) <= 0: raise ValueError('PLEASE DEFINE AT LEAST ONE HIDDENLAYER!!!')

		currentLayer = {'weights':tf.Variable(tf.random_normal([inputSize,hlSize[0]])), 'biases':tf.Variable(tf.random_normal([hlSize[0]]))}
		currentLinearModel = tf.add(tf.matmul(data,currentLayer['weights']),currentLayer['biases'])
		currentLinearModel = tf.nn.relu(currentLinearModel)

		for i in range(1,len(hlSize)):
			followingLayer = {'weights':tf.Variable(tf.random_normal([hlSize[i-1],hlSize[i]])), 'biases':tf.Variable(tf.random_normal([hlSize[i]]))}
			followingLinearModel = tf.add(tf.matmul(currentLinearModel,followingLayer['weights']),followingLayer['biases'])
			followingLinearModel = tf.nn.relu(followingLinearModel)

			currentLayer = followingLayer
			currentLinearModel = followingLinearModel

		outputLayer = {'weights':tf.Variable(tf.random_normal([hlSize[-1],classes])), 'biases':tf.Variable(tf.random_normal([classes]))}
		
		definedModel = tf.matmul(currentLinearModel,outputLayer['weights']) + outputLayer['biases']	

		return definedModel

	def trainModel(self,trainingSet,trainingLabels, testingSet, testingLabels, feedForwardCycles=10, batchSize=100,debugInfo=False):
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model,labels=self.inputLabel))
		optimizer = tf.train.AdamOptimizer().minimize(loss)

		#initialize the variables
		self.sess.run(tf.global_variables_initializer())

		#Cycles of feed forward + backpropagation
		for epoch in range(feedForwardCycles):
			currentLoss = 0

			i = 0
			while i < len(trainingSet):	
				start = i
				end = i+batchSize
				if end > len(trainingSet):
					end = len(trainingSet)

				batchTrainingSet = np.array(trainingSet[start:end]) 
				batchTrainingLabels = np.array(trainingLabels[start:end]) 

				_, c = self.sess.run([optimizer,loss], feed_dict={self.inputData:batchTrainingSet, self.inputLabel:batchTrainingLabels})
				currentLoss += c
				i += batchSize

			if(debugInfo):
				print('Cycle', epoch+1, '/', feedForwardCycles, 'loss:',currentLoss)

		#define the condition: The condition is true when the label has a 1 
		#on the same index where the model predicts the highest probability
		#argmax returns the index of the highest value of the array
		condition = tf.equal(tf.argmax(self.model,1), tf.argmax(self.inputLabel,1))
		
		#define the format of the accuracy
		accuracyFormat = tf.reduce_mean(tf.cast(condition,'float'))

		#calculate the accuracy
		with self.sess.as_default():	
			accuracy = accuracyFormat.eval({self.inputData:testingSet, self.inputLabel:testingLabels})
			if(debugInfo):			
				print('Accuracy', accuracy)

			return accuracy

	def saveModel(self, name):
		saver = tf.train.Saver()
		saver.save(self.sess, name)

	def loadModel(self, name):
		saver = tf.train.Saver()
		saver.restore(self.sess, name)

	def predict(self,data):	
		prediction = self.sess.run(self.model,feed_dict={self.inputData:data})
		sum = 0
		for i in range(len(prediction[0])):
			if prediction[0][i] >= 0:
				sum += prediction[0][i]
			else:
				prediction[0][i] = 0

		prediction[0] = prediction[0] / sum
			
		return prediction, np.argmax(prediction[0])
