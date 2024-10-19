# Building-Your-Own-Neural-Network
# Deep Learning Beginner Project - CIFAR-10 Image Classification

## Project Overview

This project is an entry-level deep learning project for beginners. It implements the classic ResNet architecture using PyTorch to classify images in the CIFAR-10 dataset. Through this project, you will learn how to build a complete deep learning pipeline, including data loading, model definition, training, evaluation, and prediction.

## Project Structure

- `datasets.py`: Defines the code for data loading and transformation.
- `resnet_model.py`: Contains the ResNet model built from scratch.
- `train.py`: Implements the model training process, including loss computation, backpropagation, and gradient updates.
- `evaluate.py`: Evaluates the model's performance on the test set.
- `predict.py`: Loads the trained model and makes predictions on new images.
- `vis.py`: Provides visualization tools to display training loss and accuracy curves.
- `requirements.txt`: Contains the dependencies required for the project, which can be installed using `pip install -r requirements.txt`.
- `log/`: Stores trained models.
- `loss/`: Stores loss and accuracy curves from the training process.

## Dataset

This project uses the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), which contains 60,000 32x32 color images in 10 classes, with 50,000 training images and 10,000 test images.

## How to Run the Project

1. **Clone the Repository**

   ```bash
   git clone https://github.com/litleWY/Building-First-Neural-Network.git
   cd Building-Your-Own-Neural-Network
   ```

2. **Install Dependencies**

   Install the required dependencies by running:

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model**

   Run `train.py` to train the ResNet model:

   ```bash
   python train.py --batch_size 64 --learning_rate 0.001 --epochs 20
   ```

4. **Evaluate the Model**

   Run `evaluate.py` to evaluate the model on the test set:

   ```bash
   python evaluate.py
   ```

5. **Predict Image Classes**

   Use the trained model to predict new images:

   ```bash
   python predict.py
   ```

## Project Features

- **ResNet from Scratch**: The project does not use a pre-trained model but instead builds ResNet-18 from scratch, helping you understand residual networks in deep learning.
- **Training By AppleM3**: Uses AppleM3 to train the model.
- **Training Visualization**: The `vis.py` script generates training loss and accuracy curves, saving them in the `loss/` folder to help you understand model convergence.

## Dependencies

- Python 3.6+
- PyTorch
- torchvision
- matplotlib
- numpy

## Folder Description

- `log/`: Stores the trained model parameter files.
- `loss/`: Stores loss and accuracy curves generated during training.

## Contribution Guidelines

Contributions are welcome! If you have any suggestions for improvements or encounter any issues, feel free to submit an issue or pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

Thanks to the PyTorch team and the open-source community for providing excellent tools and tutorials that made this project possible.

