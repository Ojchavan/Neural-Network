S&P 500 Neural Network for Stock Movement Prediction

This README provides documentation for the implementation of a neural network designed to predict stock movements in the S&P 500 based on historical price data.

Features

S&P 500 Tickers: Fetches the latest S&P 500 tickers directly from Wikipedia.

Historical Data Fetching: Downloads historical stock prices using Yahoo Finance.

Custom Data Preparation: Prepares data by creating sequences of stock prices and corresponding movement labels.

Neural Network Architecture: Custom implementation with one hidden layer using ReLU activation.

Training and Evaluation: Includes training, accuracy evaluation, and development set testing.

Parameter Saving: Saves trained model parameters for future use.

Requirements

Python 3.x

Libraries:

NumPy

pandas

yfinance

datetime

Neural Network Overview

The network predicts whether the stock price will go up (1) or down (0) based on the previous lookback_days of prices.

Architecture

Input Layer: Number of neurons equal to lookback_days (default is 30).

Hidden Layer: 20 neurons with ReLU activation.

Output Layer: 2 neurons with softmax activation (binary classification).

Parameter Initialization

Weights and biases are initialized randomly with small values to break symmetry:

W1 = np.random.rand(20, lookback_days) - 0.5
W2 = np.random.rand(2, 20) - 0.5

Activation Functions

ReLU: Used in the hidden layer to introduce non-linearity.

Softmax: Outputs probabilities for binary classification in the output layer.

Script Functions

Data Collection

Fetching S&P 500 Tickers

nn.get_sp500_tickers()

Retrieves the current list of S&P 500 tickers from Wikipedia.

Fetching Historical Stock Data

nn.get_stock_data(tickers)

Downloads 10 years of historical stock price data for the given list of tickers.

Data Preparation

Creating Sequences and Labels

X_train, Y_train, X_dev, Y_dev = nn.prepare_data(stock_prices)

Generates training and development datasets by creating sequences of stock prices and their corresponding labels.

Training

Train the Model

W1, b1, W2, b2 = nn.train(X_train, Y_train, alpha=0.01, iterations=10000)

Trains the neural network using gradient descent with forward and backward propagation.

Evaluation

Test on Development Set

dev_predictions = nn.predict(X_dev, W1, b1, W2, b2)
accuracy = nn.get_accuracy(dev_predictions, Y_dev)

Evaluates the model's performance on unseen data and calculates accuracy.

Save Model Parameters

np.savez('sp500_model.npz', W1=W1, b1=b1, W2=W2, b2=b2)

Saves the trained model parameters to a .npz file for reuse.

Example Workflow

Initialize the Neural Network:

nn = SP500NeuralNetwork(lookback_days=30)

Fetch S&P 500 Tickers:

tickers = nn.get_sp500_tickers()

Download Stock Data:

stock_prices = nn.get_stock_data(tickers)

Prepare Data:

X_train, Y_train, X_dev, Y_dev = nn.prepare_data(stock_prices)

Train the Model:

W1, b1, W2, b2 = nn.train(X_train, Y_train)

Evaluate on Development Set:

dev_predictions = nn.predict(X_dev, W1, b1, W2, b2)
accuracy = nn.get_accuracy(dev_predictions, Y_dev)
print(f"Development Set Accuracy: {accuracy:.4f}")

Save Model Parameters:

np.savez('sp500_model.npz', W1=W1, b1=b1, W2=W2, b2=b2)

Customization

Lookback Days: Change the lookback_days parameter to adjust the number of days used for prediction.

Learning Rate and Iterations: Adjust alpha and iterations in the train method to fine-tune training.

Hidden Layer Size: Modify the number of neurons in the hidden layer by changing W1 and b1 dimensions in init_params.
