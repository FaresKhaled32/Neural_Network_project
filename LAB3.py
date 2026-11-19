import math
import random


def load_data(filename):

    X = []
    Y = []

    with open(filename, "r") as file:
        for line in file:
            values = line.strip().split(",")
            values = [float(value) for value in values]
            X.append(values[0:2])
            Y.append(values[2:4])

    return X, Y
X, Y = load_data("D:/NN_Project/Neural_Network_project/ce889_dataCollection.csv")

print("Number of samples:", len(X))
print("First input:", X[0])
print("First output:", Y[0])


def split_data(X, Y):
    data = []
    for i in range(len(X)):
        data.append([X[i], Y[i]])
    random.shuffle(data)
    total = len(data)
    train_size = int(total * 0.70)
    eval_size = int(total * 0.15)
    training_data = data[:train_size]
    evaluating_data = data[train_size:train_size + eval_size]
    testing_data = data[train_size + eval_size:]
    return training_data, evaluating_data, testing_data

def data_normalization(training_data, evaluating_data, testing_data): 
  x_min = min(sample[0][0] for sample in training_data)
  x_max = max(sample[0][0] for sample in training_data)
  y_min = min(sample[0][1] for sample in training_data)
  y_max = max(sample[0][1] for sample in training_data)
  thruster_min = min(sample[1][0] for sample in training_data)
  thruster_max = max(sample[1][0] for sample in training_data)
  turning_min = min(sample[1][1] for sample in training_data)
  turning_max = max(sample[1][1] for sample in training_data) 
  datasets = [training_data, evaluating_data, testing_data]
  for dataset in datasets:
     for sample in dataset:
        inputs = sample[0]
        targets = sample[1]
        if x_max != x_min:
           inputs[0] = (inputs[0] - x_min) / (x_max - x_min)
        else:
           inputs[0] = 0.0

        if y_max != y_min:
           inputs[1] = (inputs[1] - y_min) / (y_max - y_min)
        else:
           inputs[1] = 0.0
        
        if thruster_max != thruster_min:
           targets[0] = (targets[0] - thruster_min) / (thruster_max - thruster_min)
        else:
           targets[0] = 0.0

        if turning_max != turning_min:
            targets[1] = (targets[1] - turning_min) / (turning_max - turning_min)
        else:      
            targets[1] = 0.0
            
  return (training_data, evaluating_data, testing_data, [x_min, y_min], [x_max, y_max],[thruster_min, turning_min], [thruster_max, turning_max])
class Neuron:
  
    def __init__(self,num_connections_next):
        self.activation_value=0.0
        self.weights= [
        random.uniform(-1.0, 1.0) for _ in range(num_connections_next)]
        self.bias = random.uniform(-1.0, 1.0)

    def sigmoid(self, x):

     return 1.0 / (1.0 + math.exp(-x))

    def weight_multiplication(self, weight_index):
     
     return self.activation_value * self.weights[weight_index]

    def update_weights(self, learning_rate, gradient):

     for i in range(len(self.weights)):
       
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
    total_hidden = hidden_layer[hidden_neuron].bias
    for input_neuron in range(len(input_layer)):
      total_hidden += input_layer[input_neuron].weight_multiplication(hidden_neuron)
    hidden_layer[hidden_neuron].activation_value = (hidden_layer[hidden_neuron].sigmoid(total_hidden))

  for output_neuron in range(len(output_layer)):
    total_output = output_layer[output_neuron].bias
    for hidden_neuron in range(len(hidden_layer)):
      total_output += hidden_layer[hidden_neuron].weight_multiplication(output_neuron)  
    output_layer[output_neuron].activation_value = (output_layer[output_neuron].sigmoid(total_output))

  return [neuron.activation_value for neuron in output_layer ]


def backpropagation(network,targets,learning_rate):
    
    input_layer = network[0]
    hidden_layer = network[1]
    output_layer = network[2]
    output_errors = []
    for output_neuron in output_layer:
       error = output_neuron.activation_value - targets[len(output_errors)]
       local_grad = error * output_neuron.activation_value * (1-output_neuron.activation_value)
       output_errors.append(local_grad)

    hidden_errors=[]
    for hidden_neuron in hidden_layer:
       error=0
       for output_neuron in range(len(output_layer)):
          error += output_errors[output_neuron]*hidden_neuron.weights[output_neuron]
       local_grad = error * hidden_neuron.activation_value*(1-hidden_neuron.activation_value)
       hidden_errors.append(local_grad)
     
    for hidden_neuron in hidden_layer:
       for output_neuron in range(len(output_layer)): 
          hidden_neuron.weights[output_neuron] -= (learning_rate*output_errors[output_neuron]*hidden_neuron.activation_value) 

    for output_neuron in range(len(output_layer)):
       output_layer[output_neuron].bias -= (learning_rate * output_errors[output_neuron])

    for input_neuron in input_layer:
       for hidden_neuron in range(len(hidden_layer)):
          input_neuron.weights[hidden_neuron] -= (learning_rate * hidden_errors[hidden_neuron]*input_neuron.activation_value)
    
    for hidden_neuron in range(len(hidden_layer)):
       hidden_layer[hidden_neuron].bias -= (learning_rate * hidden_errors[hidden_neuron])

def RMSE(network,dataset):
   N = len(dataset)
   total_error=0
   for sample in dataset:
      inputs = sample[0]
      outputs = sample[1]
      predictions = feed_forward(network,inputs)
      for i in range(len(outputs)):
         error = predictions[i]-outputs[i]
         total_error += error ** 2
   rmse = ((1/(N*2))*total_error)**0.5
   return rmse

def network_training(network,training_data,evaluating_data,learning_rate):
   epochs = 100
   for epoch in range(epochs):
      random.shuffle(training_data)
      for sample in training_data:
        inputs = sample[0]
        outputs = sample[1]
        feed_forward(network, inputs)
        backpropagation(network, outputs, learning_rate)
      evaluation_rmse = RMSE(network, evaluating_data)
      print("Epoch:", epoch + 1,
              "Evaluation RMSE:", evaluation_rmse)
   return network

def network_testing(network,testing_data):
   testing_rmse=RMSE(network,testing_data)
   print("Testing samples:", len(testing_data))
   print("Testing RMSE:", testing_rmse)
   return testing_rmse


def save_model(network, input_min, input_max, output_min, output_max):

    with open("D:/NN_Project/Neural_Network_project/trained_model.txt", "w") as file:

        # Save normalization values
        file.write(",".join(map(str, input_min)) + "\n")
        file.write(",".join(map(str, input_max)) + "\n")
        file.write(",".join(map(str, output_min)) + "\n")
        file.write(",".join(map(str, output_max)) + "\n")

        # Save weights and biases
        for layer in network:
            for neuron in layer:

                file.write(str(neuron.bias) + "\n")

                file.write(",".join(map(str, neuron.weights)) + "\n")

    print("Model saved successfully!")



def main():
       
       X, Y = load_data(
        "D:/NN_Project/Neural_Network_project/ce889_dataCollection.csv"
      )

    # Split dataset
       training_data, evaluating_data, testing_data = split_data(X, Y)

    # Normalise dataset
       (training_data, evaluating_data, testing_data,
       input_min, input_max, output_min, output_max) = data_normalization(
        training_data, evaluating_data, testing_data
       )

       print("Normalisation completed!")

    # Create neural network
       network = create_network(8)

    # Train neural network
       network = network_training(
        network,
        training_data,
        evaluating_data,
        0.1
       )

    # Test neural network after training
       network_testing(network, testing_data)
       # Save trained model
       save_model(network, input_min, input_max, output_min, output_max)
    
if __name__ == "__main__":
    main()
