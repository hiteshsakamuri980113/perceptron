from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator to ensure reproducibility
        random.seed(1)
        
        # Initialize weights for the first layer (3 input neurons to 4 hidden neurons)
        self.synaptic_weights1 = 2 * random.random((3, 4)) - 1
        
        # Initialize weights for the second layer (4 hidden neurons to 1 output neuron)
        self.synaptic_weights2 = 2 * random.random((4, 1)) - 1

    def __sigmoid(self, x):
        # Apply the sigmoid function to normalize the input
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        # Compute the derivative of the sigmoid function
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations, learning_rate=0.1):
        # Train the neural network through a process of trial and error
        for iteration in range(number_of_training_iterations):
            # Forward pass: compute the outputs of the network
            output1, output2 = self.think(training_set_inputs)
            
            # Calculate the error at the output layer
            error = training_set_outputs - output2
            
            # Calculate the adjustment for the weights between the hidden layer and the output layer
            adjustment2 = dot(output1.T, error * self.__sigmoid_derivative(output2)) * learning_rate
            
            # Calculate the error at the hidden layer
            error1 = dot(error * self.__sigmoid_derivative(output2), self.synaptic_weights2.T)
            
            # Calculate the adjustment for the weights between the input layer and the hidden layer
            adjustment1 = dot(training_set_inputs.T, error1 * self.__sigmoid_derivative(output1)) * learning_rate
            
            # Update the weights with the calculated adjustments
            self.synaptic_weights1 += adjustment1
            self.synaptic_weights2 += adjustment2

    def think(self, inputs):
        # Forward pass: compute the output of the hidden layer
        output1 = self.__sigmoid(dot(inputs, self.synaptic_weights1))
        
        # Forward pass: compute the output of the output layer
        output2 = self.__sigmoid(dot(output1, self.synaptic_weights2))
        
        # Return the outputs of both layers
        return output1, output2

if __name__ == "__main__":
    # Create an instance of the neural network
    neural_network = NeuralNetwork()
    
    # Print the initial random weights
    print("Random starting synaptic weights (layer 1): ")
    print(neural_network.synaptic_weights1)
    print("Random starting synaptic weights (layer 2): ")
    print(neural_network.synaptic_weights2)

    # Define the training set inputs and outputs
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 0]])
    training_set_outputs = array([[0, 1, 1, 0, 0]]).T

    # Train the neural network
    neural_network.train(training_set_inputs, training_set_outputs, 100000, learning_rate=0.1)

    # Print the new weights after training
    print("New synaptic weights after training (layer 1): ")
    print(neural_network.synaptic_weights1)
    print("New synaptic weights after training (layer 2): ")
    print(neural_network.synaptic_weights2)

    # Test the neural network with new situations
    print("Considering new situation [1, 0, 0] -> ?: ")
    print(neural_network.think(array([1, 0, 0]))[1])

    print("Considering new situation [0, 1, 0] -> ?: ")
    print(neural_network.think(array([0, 1, 0]))[1])

    print("Considering new situation [1, 1, 0] -> ?: ")
    print(neural_network.think(array([1, 1, 0]))[1])