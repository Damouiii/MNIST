import torch

# Import the torch library, which is a popular deep learning framework

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Check if a CUDA-enabled GPU is available using torch.cuda.is_available()
# If GPU is available, set device to 'cuda' to run code on GPU
# If GPU is not available, set device to 'cpu' to run code on CPU

# Print the selected device ('cuda' or 'cpu')
print(device)

# Import the necessary modules
# Import the datasets module from torchvision library
# The torchvision library provides popular datasets like MNIST, CIFAR10, etc.
# These datasets can be used for computer vision tasks
from torchvision import datasets

# Import the ToTensor class from the transforms module in torchvision
# The ToTensor transform is used to convert input data, such as images, into PyTorch tensors
# It ensures that the data is in the appropriate format for deep learning tasks
from torchvision.transforms import ToTensor


# Import the necessary modules
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download the MNIST training dataset
train_data = datasets.MNIST(
    root='data',              # Root directory where the dataset will be stored
    train=True,               # Download the training split of the dataset
    transform=ToTensor(),     # Convert the image to a PyTorch tensor
    download=True,            # Download the dataset if it doesn't exist
)

# Download the MNIST test dataset
test_data = datasets.MNIST(
    root='data',              # Root directory where the dataset will be stored
    train=False,              # Download the test split of the dataset
    transform=ToTensor(),     # Convert the image to a PyTorch tensor
)

# Print information about the training dataset
print(train_data)
print(train_data.data.size())       # Size/shape of the input images
print(train_data.targets.size())    # Number of labels in the training dataset
# Preparing the data for training
from torch.utils.data import DataLoader

# Import the DataLoader class from torch.utils.data module
# The DataLoader class provides an iterable over a dataset for batching and parallel loading

# The MNIST (Modified National Institute of Standards and Technology) data consists of 60,000 training images
# and 10,000 test images. Each image is a crude 28 x 28 (784 pixels) handwritten digit from "0" to "9."
# Each pixel value is a grayscale integer between 0 and 255.

batch_size = 100
# Set the batch size for training and testing data
# A batch is a subset of the dataset that is processed together during training or inference

loaders = {
    'train': torch.utils.data.DataLoader(train_data,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=1),
    # Create a DataLoader object named 'train' for the training data
    # Pass the 'train_data' dataset to the DataLoader
    # Set the batch size using the 'batch_size' variable
    # Set shuffle=True to randomize the order of the data during training
    # Set num_workers=1 to use a single worker for loading the data

    'test': torch.utils.data.DataLoader(test_data,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=1),
    # Create a DataLoader object named 'test' for the testing data
    # Pass the 'test_data' dataset to the DataLoader
    # Set the batch size using the 'batch_size' variable
    # Set shuffle=True to randomize the order of the data during testing
    # Set num_workers=1 to use a single worker for loading the data
}
import torch.nn as nn

# Import the nn module from torch to define neural network models

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Define the CNN model by inheriting from the nn.Module class

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # Define the first convolutional layer (conv1) using nn.Sequential
        # nn.Conv2d creates a 2D convolutional layer
        # in_channels=1 specifies that the input has 1 channel (grayscale image)
        # out_channels=16 specifies the number of output channels or filters
        # kernel_size=5 defines the size of the convolutional kernel (5x5)
        # stride=1 specifies the stride of the convolution operation
        # padding=2 adds padding to the input to ensure that the spatial dimensions remain the same after convolution
        # nn.ReLU() applies the ReLU activation function
        # nn.MaxPool2d performs max pooling with a kernel size of 2 and stride of 2 to downsample the feature maps

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Define the second convolutional layer (conv2) using nn.Sequential
        # nn.Conv2d creates a 2D convolutional layer
        # The input has 16 channels (output channels from conv1)
        # out_channels=32 specifies the number of output channels or filters
        # kernel_size=5 defines the size of the convolutional kernel (5x5)
        # stride=1 specifies the stride of the convolution operation
        # padding=2 adds padding to the input to ensure that the spatial dimensions remain the same after convolution
        # nn.ReLU() applies the ReLU activation function
        # nn.MaxPool2d performs max pooling with a kernel size of 2 and stride of 2 to downsample the feature maps

        self.out = nn.Linear(32 * 7 * 7, 10)
        # Define the fully connected layer (out) using nn.Linear
        # The input size is 32 * 7 * 7, which is the number of output channels from conv2 multiplied by the spatial dimensions of the feature maps (7x7)
        # The output size is 10, representing the number of classes in the classification task

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # Perform forward pass through conv1 and conv2

        x = x.view(x.size(0), -1)
        # Flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        # The size of the first dimension (batch_size) remains the same, and the second dimension is flattened

        output = self.out(x)
        # Perform forward pass through the fully connected layer (out) to obtain the output logits

        return output, x
        # Return the output logits and x (flattened feature maps) for visualization purposes


def train(num_epochs, cnn, loaders):
    # Function to train the CNN model

    from torch.autograd import Variable
    from torch import optim

    # Import necessary modules for training

    optimizer = optim.Adam(cnn.parameters(), lr=0.01)
    # Define the optimizer (Adam) and specify the learning rate
    # Pass the parameters of the CNN model (cnn.parameters()) to the optimizer for optimization

    loss_func = nn.CrossEntropyLoss()
    # Define the loss function (CrossEntropyLoss) for classification tasks

    cnn.train()
    # Set the model to train mode

    total_step = len(loaders['train'])
    # Calculate the total number of batches in the training data

    for epoch in range(num_epochs):
        # Iterate over the specified number of epochs

        for i, (images, labels) in enumerate(loaders['train']):
            # Iterate over the batches of the training data

            b_x = Variable(images)  # batch x
            b_y = Variable(labels)  # batch y

            # Convert the batched input images and labels to Variables
            # Variables are deprecated in recent PyTorch versions and can be replaced with tensors

            output = cnn(b_x)[0]
            # Perform forward pass through the CNN model to obtain the output logits

            loss = loss_func(output, b_y)
            # Compute the loss by comparing the output logits with the ground truth labels

            optimizer.zero_grad()
            # Clear the gradients of the optimizer for this training step

            loss.backward()
            # Perform backpropagation to compute the gradients of the loss with respect to the model parameters

            optimizer.step()
            # Update the model parameters by applying the computed gradients using the optimizer

            if (i + 1) % 100 == 0:
                # Print the training progress every 100 batches

                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    PATH = "mnist_trained_model.pt"
    # Specify the file path to save the trained model

    torch.save(cnn.state_dict(), PATH)
    # Save the state dictionary of the CNN model to the specified file path


def test():
    # Function to test the trained model on the test dataset

    # Load the trained model
    PATH = "mnist_trained_model.pt"
    cnn = CNN()
    cnn.load_state_dict(torch.load(PATH))
    cnn.eval()
    # Load the saved state dictionary of the CNN model
    # Set the model to evaluation mode using the eval() method

    # Test the model
    with torch.no_grad():
        # Disable gradient calculation since we are only testing

        correct = 0
        total = 0
        final_accuracy = 0

        total_iterations = len(loaders['test'].dataset.data) / batch_size
        # Calculate the total number of iterations (batches) in the test dataset

        for images, labels in loaders['test']:
            # Iterate over the batches of the test dataset

            test_output, last_layer = cnn(images)
            # Perform forward pass through the CNN model to obtain the test output logits

            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            # Get the predicted labels by finding the maximum value along the second dimension of the test output logits

            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            # Compute the accuracy by comparing the predicted labels with the ground truth labels

            final_accuracy += accuracy
            # Accumulate the accuracy for each batch

            print("The current batch accuracy: {}".format(accuracy))

    print("The current accuracy of {} images is {}".format(len(loaders['test'].dataset.data), final_accuracy / total_iterations))
    # Print the overall accuracy of the model on the test dataset

if __name__=="__main__":
    cnn = CNN()
    num_epochs = 10
    train(num_epochs, cnn, loaders) # This line runs only once.
    #test()

