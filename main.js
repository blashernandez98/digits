const DeepNeuralNetwork = require('./dnn')
const math = require('mathjs')

// Example dataset (X = input features, Y = target labels)
const X = math.matrix([
  [0, 0, 1, 1],
  [0, 1, 0, 1],
  [1, 1, 1, 0],
])

const Y = math.matrix([[0, 1, 1, 0]])

// Create a DNN with 3 input neurons (nx = 3) and 2 hidden layers with 4 and 1 neurons
const dnn = new DeepNeuralNetwork(3, [4, 1])

// Perform forward propagation
const predictions = dnn.forwardProp(X)
console.log('Predictions:', predictions)

// Perform gradient descent
dnn.gradientDescent(Y)

// Save weights to a JSON string
const savedWeights = dnn.saveWeights()
console.log('Saved Weights:', savedWeights)

// Load weights from a JSON string
dnn.loadWeights(savedWeights)

// Evaluate the model
const [predLabels, cost] = dnn.evaluate(X, Y)
console.log('Predictions:', predLabels)
console.log('Cost:', cost)
