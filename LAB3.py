import math
import random

class Neuron:
  
    def __init__(self,num_connections_next):
        self.activation_value=0.0
        self.weights= [
        random.uniform(-1.0, 1.0) for _ in range(num_connections_next)]

    def sigmoid(self, x):

     return 1.0 / (1.0 + math.exp(-x))

    def weight_multiplication(self, weight_index):
     
     return self.activation_value * self.weights[weight_index]

    def update_weights(self, learning_rate, gradient):

     for i in range(len(self.Weights)):
       
       self.weights[i] -= learning_rate * gradient[i]    

def create_network(hidden_size):
      
      input_layer = [Neuron(hidden_size), Neuron(hidden_size) ]

      hidden_layer = [Neuron(2) for _ in range(hidden_size)]

      output_layer = [Neuron(0),Neuron(0)]

      network = [input_layer,hidden_layer,output_layer ]

      return network

def feed_forward(network,inputs):
  input_layer = network[0]
  hidden_layer = network[1]
  output_layer = network[2]

  for i in range(len(input_layer)):
    input_layer[i].activation_value=inputs[i]

  for hidden_neuron in range(len(hidden_layer)):
    total_hidden = 0
    for input_neuron in range(len(input_layer)):
      total_hidden += input_layer[input_neuron].weight_multiplication(hidden_neuron)
    hidden_layer[hidden_neuron].activation_value = (hidden_layer[hidden_neuron].sigmoid(total_hidden))

  for output_neuron in range(len(output_layer)):
    total_output = 0
    for hidden_neuron in range(len(hidden_layer)):
      total_output += hidden_layer[hidden_neuron].weight_multiplication(output_neuron)  
    output_layer[output_neuron].activation_value = (output_layer[output_neuron].sigmoid(total_output))

  return [neuron.activation_value for neuron in output_layer ]

def main():
  

    network = create_network(8)

    inputs = [247.15, 575.9]

    predictions = feed_forward(network, inputs)

    print("Thruster:", predictions[0])
    print("Turning:", predictions[1])
if __name__ == "__main__":
    main()
