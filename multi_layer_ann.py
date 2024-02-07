import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import csv
import os
from tqdm import trange

# Get the current working directory path
current_directory = os.getcwd()

# Neural network class definition.
class neuralNetwork:   
    
    # Initialise the neural network.
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate, dropout_rate=None, noise_level=None, wkj=None,wjm=None,bj=None,bm=None):
        
        # Set number of nodes in each input, hidden, output layer.
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes

        self.wkj = wkj
        self.wjm = wjm
        self.bj = bj
        self.bm = bm

        if (wkj is None and wjm is None and bj is None and bm is None):   
            self.wkj = np.random.uniform(size=self.inodes * self.hnodes, low=-1, high=1).reshape(self.hnodes, self.inodes)
            self.wjm = np.random.uniform(size=self.hnodes * self.onodes, low=-1, high=1).reshape(self.hnodes,self.onodes)        

            self.bj = np.random.uniform(size=self.hnodes, low=-1, high=1).reshape(self.hnodes, 1)
            self.bm = np.random.uniform(size=self.onodes, low=-1, high=1).reshape(self.onodes, 1)
        
        # Learning rate
        self.lr = learningRate

        # Activation mode
        self.activation_mode = None

        # Dropout layer
        self.dropout_rate = dropout_rate

        # Noise level
        self.noise_level = noise_level

        # Used for testing neural network's success.
        self.actual_list = np.array([])
        self.predict_list = np.array([])

        # Used for calculating MSE (Mean Squared Error).
        self.error_list = np.array([])

    # Activation function is;     
    #   Si for sigmoid, St for step,
    #   Re for relu, Ht for hyperbolic tangent.
    def activation_func(self, x, mod = None):

        # Stores the "mod" variable so that the net derivative calculation 
        # based on activation functions can be calculated in the "train" function.
        self.activation_mode = mod

        if mod == 'Si' or mod == None:
            return 1 / (1 + np.exp(-x))
        elif mod == 'St':
            return np.array(x > 0, dtype=np.float32)
        elif mod == 'Re':
            return np.maximum(x,0)
        elif mod == 'Ht':
            return np.tanh(x)
        
    # Train the neural network.
    def train(self, input_list, target_list):

        ###### Forward propagation ######

        # Convert input list to 2d array.
        inputs = np.array(input_list, ndmin=2).T

        # Apply noise to the input layer if noise_level is specified
        if noise_level is not None:
            noise = np.random.normal(0, noise_level, inputs.shape)
            inputs = inputs + noise

        # Calculate signals into hidden layer.
        Net_j = np.dot(self.wkj, inputs) + self.bj
        Output_j = self.activation_func(Net_j)

        # Apply dropout to the hidden layer if dropout_rate is specified
        if self.dropout_rate is not None:
            dropout_mask = (np.random.rand(*Output_j.shape) < self.dropout_rate) / (1 - self.dropout_rate)
            Output_j *= dropout_mask

        # Calculate signals into output layer.
        Net_m = np.transpose(np.dot(np.transpose(Output_j),self.wjm)) + self.bm
        Output_m = self.activation_func(Net_m)

        Expected_output = np.array(target_list, ndmin=2).T

        ###### Backpropagation ######

        ###  Output layer's weights calculation

        Error_m = Expected_output - Output_m

        # Save predicted and expected results in lists.
        self.error_list = np.append(self.error_list, Error_m)
        
        delta_m = Output_m * (1 - Output_m) * Error_m
        delta_Wjm = np.dot(Output_j, self.lr * np.transpose(delta_m))  
        self.wjm += delta_Wjm

        delta_Bm = self.lr * delta_m
        self.bm += delta_Bm

        ###  Hidden layer's weights calculation

        hidden_error = np.dot(self.wjm, delta_m)

        delta_j = Output_j * (1 - Output_j) * hidden_error 
        delta_Wkj = np.dot(self.lr * delta_j, np.transpose(inputs))
        self.wkj += delta_Wkj

        delta_Bj = self.lr * delta_j
        self.bj += delta_Bj

    # Predict output according to neural network's weights.
    # Basically make forward propagation and return output layer's outputs.
    def predict(self, input_list):

        # Convert input list to 2d array.
        inputs = np.array(input_list, ndmin=2).T

        # Calculate signals into hidden layer.
        Net_j = np.dot(self.wkj, inputs) + self.bj
        Output_j = self.activation_func(Net_j)

        # Calculate signals into output layer
        Net_m = np.transpose(np.dot(np.transpose(Output_j),self.wjm)) + self.bm
        Output_m = self.activation_func(Net_m)

        return Output_m
    
    # Normalize values using lineer interpolation
    def normalize(self, original_range, desired_range, input):
        
        original_range = np.array(original_range)
        desired_range = np.array(desired_range)

        # Calculation of lineer interpolation 
        normalized_input = np.interp(input, original_range, desired_range)

        return normalized_input

# Class definition which include all showing and saving methods inside.
class ResultPrinter:
    def __init__(self):
        self.epochs = []
        self.mse_values = []

    # Calculare MSE (Mean Squared Error).
    def calculate_mse(self, show=False, epoch=None, n=None, error=None):
        if show:
            self.show_mse()
        else:
            self.epochs.append(epoch)
            error_square = error **2 
            mse_value = sum(error_square) / n  # Assuming error is a list of squared errors.
            self.mse_values.append(mse_value)
    
    # Plots the MSE (Mean Squared Error).
    def show_mse(self):
        plt.plot(self.epochs, self.mse_values)
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title('MSE over Epochs')
        plt.show()

    # This function prints and plots the confusion matrix.
    # Normalization can be applied by setting `normalize=True`.
    def show_confusion_matrix(x_expected, y_predicted, normalize=False, title=None, cmap=plt.cm.Blues):
        if not title:
            title = 'Confusion matrix'

        # Compute confusion matrix.
        cm = confusion_matrix(x_expected, y_predicted)
        classes = np.unique(np.concatenate([x_expected, y_predicted]))

        if normalize:
            # Normalize the confusion matrix if specified.
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        """print(cm)"""

        # Create a new figure and axis for plotting.
        fig, ax = plt.subplots()
        
        # Display the confusion matrix as an image.
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        
        # Set axis labels and tick marks.
        ax.set(xticks=np.arange(len(classes)),
            yticks=np.arange(len(classes)),
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='Expected Value',
            xlabel='Predicted Value')

        # Rotate the tick labels for better visibility.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations for each cell.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
                
                # Check if the prediction is correct, and if so, add a light green border.
                if i == j and cm[i, j] > 0:
                    # Add a rectangle with a light green border around correctly predicted cells.
                    rect = plt.Rectangle((j - 0.5 + 0.05, i - 0.5 + 0.05), 0.9, 0.9, fill=False, edgecolor='lime', linewidth=5, alpha=1)
                    ax.add_patch(rect)

        # Adjust layout for better appearance.
        fig.tight_layout()
        
        # Return the figure and axis.
        return fig, ax
    
    # Save weights as CSV file.
    def print_weights(file):
        try:
            # Create a CSV file and write the data.
            with open(file, 'w', newline='') as f:
                # Write the data sequentially to the file.
                writer = csv.writer(f)
                
                # Write the weights from input to hidden layer (wkj).
                writer.writerow(['wkj:'])
                np.savetxt(f, n.wkj, delimiter=',')
                
                # Write the weights from hidden to output layer (wjm).
                writer.writerow(['wjm:'])
                np.savetxt(f, n.wjm, delimiter=',')
                
                # Write the biases for the hidden layer (bj).
                writer.writerow(['bj:'])
                np.savetxt(f, n.bj, delimiter=',')
                
                # Write the biases for the output layer (bm).
                writer.writerow(['bm:'])
                np.savetxt(f, n.bm, delimiter=',')
        except Exception as e:
            print(f"Error: {e}")

    # Read weights from CSV file.
    def read_weights(file_path):
        # Initialize empty lists to store weights for different sections
        wkj, wjm, bj, bm = [], [], [], []

        # Open the CSV file for reading
        with open(file_path, 'r') as file:
            # Create a CSV reader object
            reader = csv.reader(file)
            current_section = None  # Variable to keep track of the current section

            # Loop through each row in the CSV file
            for row in reader:
                if row:
                    # Check if the row ends with a colon (indicating a new section)
                    if row[0].endswith(":"):
                        current_section = row[0].strip(":").lower()
                    else:
                        # Convert the values in the row to floats and store them in the corresponding section
                        weights = np.array([float(value) for value in row])
                        if current_section == "wkj":
                            wkj.append(weights)
                        elif current_section == "wjm":
                            wjm.append(weights)
                        elif current_section == "bj":
                            bj.append(weights)
                        elif current_section == "bm":
                            bm.append(weights)

        # Convert the lists to NumPy arrays and return them
        return np.array(wkj), np.array(wjm), np.array(bj), np.array(bm)

# Initialise neural network parameters.

input_nodes = 784
hidden_nodes = 196
output_nodes = 10
learning_rate = 0.1
dropout_rate = None
noise_level = None
# dropout_rate = 0.10
# noise_level = 0.03

# Create instance of neural network.
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate, dropout_rate, noise_level)

# If there are previously trained weight file exist

"""
relative_pre_trained_weight_path = "Pre_trained_weights/weight_bias_value_750.csv"
pre_trained_weight_path = os.path.join(current_directory, relative_pre_trained_weight_path)

wkj, wjm, bj, bm = ResultPrinter.read_weights(pre_trained_weight_path)

# Create an instance of the neural network.
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate, dropout_rate, noise_level, wkj, wjm, bj, bm)"""


# Create instance of result printer.
result_printer = ResultPrinter()

### Train the neural network.
###########################################################################

relative_training_path = "Data/fashion-mnist_train.csv"
training_path = os.path.join(current_directory, relative_training_path)

training_data_file = open(training_path, 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# Epochs is the number of times the training data set is used for training.
epochs = 1001

# Get the number of data points.
num_data_points = len(training_data_list)

for epoch in trange(epochs):
    
    # Go through all records in the training data set.
    for record in training_data_list:
        
        # Split the record by the ',' commas.
        all_values = record.split(',')

        # Min and max values for original input and normalized input.
        original_range = [0, 255]
        desired_range = [0.01, 0.99]

        # Normalized inputs
        inputs = n.normalize(original_range,desired_range,np.asfarray(all_values[1:]))

        # Create the target output values (all 0.01, except the desired label which is 0.99).
        Expected_output = np.zeros(output_nodes) + 0.01
       
        # all_values[0] is the target label for this record.
        Expected_output[int(all_values[0])] = 0.99
        n.train(inputs, Expected_output)

    # Calculate MSE for the current epoch and add it to the result_printer.
    result_printer.calculate_mse(epoch=epoch, n=num_data_points, error=n.error_list)
    n.error_list = np.array([]) # Reset error_list after this epoch MSE calculated.

    if ((epoch%50 == 0) and epoch!=0):
        # Save weights
        weight_path = os.path.join(current_directory, "weight_bias_value_"+str(epoch)+".csv")
        ResultPrinter.print_weights(weight_path)       

# Display the MSE plot after training.
result_printer.calculate_mse(show=True)
###########################################################################

### Test the neural network.
###########################################################################

relative_test_path = "Data/fashion-mnist_test.csv"
test_path = os.path.join(current_directory, relative_test_path)

test_data_file = open(test_path, 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
prediction_count = 0

for record in test_data_list:

    # Split the record by the ',' commas.
    all_values = record.split(',')

    # Min and max values for original input and normalized input.
    original_range = [0, 255]
    desired_range = [0.01, 0.99]

    # Normalized inputs
    inputs = n.normalize(original_range,desired_range,np.asfarray(all_values[1:]))

    # Calculate predicted list. 
    predicted_result = n.predict(inputs)

    # Save predicted and expected results in lists.
    n.predict_list = np.append(n.predict_list, np.argmax(predicted_result))
    n.actual_list = np.append(n.actual_list, int(all_values[0]))

    # Show predicted an expected results in test set.
    """print("Predicted Output: %d, Expected Output: %d " % (np.argmax(predicted_result), int(all_values[0])))"""

    # Calculate total correct prediction count.
    if np.argmax(predicted_result) == int(all_values[0]):
        prediction_count += 1    
    
# Calculate success rate of neural network.
print("Success: %f" % (prediction_count / len(test_data_list)))
###########################################################################

### Show confusion_matrix
###########################################################################
x_expected = n.actual_list
y_predicted = n.predict_list

ResultPrinter.show_confusion_matrix(x_expected=x_expected, y_predicted=y_predicted, normalize=False)
plt.show()
###########################################################################



