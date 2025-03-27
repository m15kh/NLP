# Simple RNN for Sentiment Analysis

This project implements a Recurrent Neural Network (RNN) for sentiment analysis on the IMDB movie review dataset using TensorFlow and Keras.

## Overview

The notebook (`simple_RNN.ipynb`) demonstrates:
- Basic usage of SimpleRNN layer in TensorFlow
- Processing and preparing the IMDB dataset for sentiment analysis
- Building and training a sequential model with embedding and RNN layers
- Evaluating model performance on test data

## Model Architecture
 
The model consists of:
1. Embedding layer (converts words to dense vectors of fixed size)
2. SimpleRNN layer (processes sequential data and captures temporal dependencies)
3. Dense output layer with sigmoid activation (for binary classification)

## Dataset

The IMDB dataset contains 25,000 movie reviews for training and 25,000 for testing, labeled as positive or negative.

## Requirements

- TensorFlow 2.x
- Keras
- NumPy

## Usage

1. Open the notebook in Google Colab or Jupyter
2. Execute cells sequentially to:
   - Load and preprocess the IMDB dataset
   - Define the RNN model
   - Train the model
   - Evaluate performance

## Results

The model achieves approximately 70% accuracy on sentiment classification after training for 15 epochs.

## Model Parameters

- Maximum features (vocabulary size): 20,000
- Maximum sequence length: 80
- Embedding dimension: 128
- RNN units: 128
- Batch size: 32
- Training epochs: 15
