# aigo
AIGO is Deep Learning Architecture based on Golang.

This was originally a school project, I think it's good to share anybody who wants to implement deep learning architecture using golang



## Prerequisite

\>= Go 1.13.4 

data files from http://yann.lecun.com/exdb/mnist/

Please locate data file with same location with main.go

  \- t10k-images.idx3-ubyte: Decompressed file of t10k-images.idx3-ubyte.gz

  \- t10k-labels.idx1-ubyte: Decompressed file of t10k-labels.idx1-ubyte.gz

  \- train-images.idx3-ubyte: Decompressed file of train-images.idx3-ubyte.gz

  \- train-labels.idx1-ubyte: Decompressed file of train-labels.idx1-ubyte.gz



## Execution

Just 

```
$ go build
```

and execute. It will train a basic FC Network for MNIST.



## Features

- ReLU Activation Function Implementation (Actually it's ReLU6)
- LeakyReLU Activation Function Implementation (Actually it's LeakyReLU6)
- Softmax Activation Function Implementation
- Linear Fully Connection Layer Implementation
- Calculation of CrossEntropyLoss
- Data Loading of the MNIST Dataset
- Outputs log file
- Outputs top N images of the specific label.

## Result

### ReLU 784-512-512-256-128-10

For basic setup (Epoch = 4, Learning Rate 0.0001), final test accuracy is 92.51 (I know it's quite low...) Execution Time was 1h39m27.5877381s

### LeakyReLU 784-512-512-256-128-10

For basic setup (Epoch = 4, Learning Rate 0.0001), final test accuracy is 92.516 (I know it's quite low... too...) Execution Time was 1h41m36.5955387s

## Maybe todo? but idk

Using interface (**again**) to make code better and cleaner.

Allowing input of 2d array, and convert it to 1d array inside layer. - This will help implementation of Convolutional layers

Implement Convolutional layers