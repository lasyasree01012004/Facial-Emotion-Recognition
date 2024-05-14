# Facial Expression Recognition with Deep Learning Approaches

This repository contains three different approaches for facial expression recognition using deep learning techniques implemented in PyTorch. The approaches are:

1. **Deep Convolutional Neural Network (CNN)**
2. **Pre-trained Models (ResNet50, VGG16, MobileNet)**
3. **Ensemble Model (Deep CNN + GCN)**
4. **Graph Convolutional Network (GCN)**

## Dependencies

- Python 3.x
- PyTorch
- torchvision
- Matplotlib
- Pillow
- NumPy
- Pandas
- tqdm
- OpenCV
- scikit-learn

## Usage

1. The dataset is loaded usign kaggle to the directory.
2. Run the code to train and evaluate the models.

## Model Architectures

1. **Deep CNN**: The code defines a `Deep_Emotion` class that inherits from `nn.Module`. The model architecture consists of convolutional layers, max pooling, batch normalization, and fully connected layers.
2. **Ensemble Model**: The ensemble model combines the predictions from the Deep CNN and GCN models using a weighted average or other ensemble technique.
3. **Graph Convolutional Network (GCN)**: The code defines a `GraphConvolutionalNetwork` class that inherits from `nn.Module`. The model architecture consists of Graph Convolution layers, Global Average Pooling, fully connected layers, ReLU activation, and Softmax output.
   
## Training and Evaluation

The code trains the Deep CNN and GCN models using PyTorch's `DataLoader` and `TensorDataset`. The training loops iterate over the specified number of epochs and update the model parameters using optimization algorithms like Adam with weight decay and label smoothing.

After training, the GCN model's state dictionary is saved to a file named `fer13_gcn_weights.pth` in the `model_dir` directory.

The code then loads the saved GCN model and evaluates its performance on the test set, printing the accuracy and other relevant metrics.

Here, pre-trained models are also used for comparision
