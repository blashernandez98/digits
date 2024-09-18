const math = require('mathjs')

function sigmoid(z) {
  return math.map(z, (val) => 1 / (1 + Math.exp(-val)))
}

function softmax(z) {
  // Subtract the maximum value in each column for numerical stability
  const zMax = math.max(z, 0)
  const expZ = math.map(z, (value) => Math.exp(value - zMax))

  // Calculate sum of exponentials along columns
  const sumExpZ = math.sum(expZ, 0)

  // Normalize exponentials
  return math.dotDivide(expZ, sumExpZ)
}

class DeepNeuralNetwork {
  constructor(nx, layers) {
    // Validate input
    if (!Number.isInteger(nx) || nx < 1) {
      throw new Error('nx must be a positive integer')
    }
    if (
      !Array.isArray(layers) ||
      layers.length < 1 ||
      layers.some((x) => !Number.isInteger(x) || x < 1)
    ) {
      throw new Error('layers must be a list of positive integers')
    }

    this.L = layers.length // Number of layers
    this.cache = {}
    this.weights = {}

    // Initialize weights and biases using He initialization
    for (let i = 0; i < this.L; i++) {
      if (i === 0) {
        this.weights[`W${i + 1}`] = math.multiply(
          math.random([layers[i], nx], -1, 1),
          Math.sqrt(2 / nx)
        )
      } else {
        this.weights[`W${i + 1}`] = math.multiply(
          math.random([layers[i], layers[i - 1]], -1, 1),
          Math.sqrt(2 / layers[i - 1])
        )
      }
      this.weights[`b${i + 1}`] = math.zeros([layers[i], 1])
    }
  }

  // Forward propagation
  forwardProp(X) {
    this.cache['A0'] = X
    for (let i = 1; i <= this.L; i++) {
      let W = this.weights[`W${i}`]
      let b = this.weights[`b${i}`]
      let A_prev = this.cache[`A${i - 1}`]

      // Z = W * A_prev + b
      let Z = math.add(math.multiply(W, A_prev), b)

      // Activation
      if (i === this.L) {
        // Use softmax for the output layer
        this.cache[`A${i}`] = softmax(Z)
      } else {
        // Use sigmoid for hidden layers
        this.cache[`A${i}`] = sigmoid(Z)
      }
    }
    return this.cache[`A${this.L}`]
  }

  // Compute cost (cross-entropy loss)
  cost(Y, A) {
    const m = Y[0].length // Number of samples
    const logprobs = math.dotMultiply(Y, math.log(A))
    return -math.sum(logprobs) / m
  }

  // Evaluate the network's predictions
  evaluate(X, Y) {
    let predictions = this.forwardProp(X)
    let cost = this.cost(Y, predictions)

    // Return binary predictions (1 if A > 0.5 else 0) and the cost
    let predLabels = predictions.map((pred) =>
      pred.map((val) => (val > 0.5 ? 1 : 0))
    )
    return [predLabels, cost]
  }

  gradientDescent(Y, alpha = 0.05) {
    const m = Y[0].length // Number of samples
    let dz = math.subtract(this.cache[`A${this.L}`], Y)

    for (let i = this.L; i > 0; i--) {
      const W_i = this.weights[`W${i}`]
      const A_prev = this.cache[`A${i - 1}`]

      // Calculate dw and db
      const dw = math.multiply(dz, math.transpose(A_prev))
      const db = math.sum(dz, 1)

      // Update dz for the next iteration
      if (i > 1) {
        const A_prev_sigmoid = this.cache[`A${i - 1}`]
        dz = math.dotMultiply(math.transpose(W_i), dz)
        dz = math.dotMultiply(
          dz,
          math.dotMultiply(A_prev_sigmoid, math.subtract(1, A_prev_sigmoid))
        )
      }

      // Update weights and biases
      this.weights[`W${i}`] = math.subtract(
        this.weights[`W${i}`],
        math.multiply(alpha, math.divide(dw, m))
      )
      this.weights[`b${i}`] = math.subtract(
        this.weights[`b${i}`],
        math.multiply(alpha, math.divide(db, m))
      )
    }
  }

  // Save weights to JSON file
  saveWeights() {
    return JSON.stringify(this.weights)
  }

  // Load weights from JSON
  loadWeights(weightsJSON) {
    this.weights = JSON.parse(weightsJSON)
  }
}

module.exports = DeepNeuralNetwork
