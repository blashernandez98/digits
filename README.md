# Handwritten Digit Classifier - Deep Neural Network

This project is a simple implementation of a Deep Neural Network (DNN) in JavaScript using the `mathjs` library. The network is designed to classify handwritten digits, inspired by the famous MNIST dataset. It is built from scratch, including forward propagation, backpropagation, and gradient descent for training.

## Features

- **Deep Neural Network**: Supports multiple layers of neurons, with customizable layer sizes.
- **Sigmoid and Softmax Activation**: Implements sigmoid activation for hidden layers and softmax for the output layer.
- **Cross-Entropy Loss**: Utilizes cross-entropy as the loss function.
- **Gradient Descent**: Includes backpropagation with gradient descent for model training.

## Files

- **`dnn.js`**: Contains the class definition for the deep neural network.
- **`main.js`**: Demonstrates how to use the neural network for digit classification.

## Getting Started

### Prerequisites

- **Node.js**: Ensure that Node.js is installed on your system.
- **mathjs**: This library is required for matrix and mathematical operations.

To install `mathjs`, run:

```bash
npm install mathjs
```

### Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/blashernandez98/digits.git
   cd digit-classifier-dnn
   ```

2. **Create a DNN instance and run it:**

   The following example in `main.js` shows how to use the `DeepNeuralNetwork` class:

   ```javascript
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
   ```

3. **Run the code**:

   ```bash
   node main.js
   ```

### Example Output

```
Predictions: [[0.5], [0.6], [0.3]]
Saved Weights: {"W1":...,"b1":...}
Predictions: [[1], [0], [1]]
Cost: 0.321
```

## License

This project is licensed under the MIT License.
