# Predicting Stress-Strain Curves from Indentation Data using Deep Learning

This project implements a deep learning approach to predict stress-strain curves of irradiated Eurofer 97 steel using indentation (load-displacement) data. The model uses a hybrid neural network architecture combining LSTM and dense layers to establish relationships between indentation behavior and mechanical properties.

## Overview

The project processes and analyzes two main types of experimental data:
- Load-displacement (PH) curves from indentation tests
- Stress-strain (SS) curves from tensile tests

Data is collected at various temperatures (250°C, 300°C, 350°C, 400°C, 450°C, 500°C) including room temperature conditions (RT) to understand temperature-dependent material behavior.

## Features

- Data preprocessing pipeline for both PH and SS curves including:
  - Origin shift normalization
  - Cutoff-based filtering
  - Data smoothing using averaging windows
  - Feature scaling using StandardScaler
- Neural network architecture combining:
  - LSTM layers for sequential data processing
  - Dense layers for feature extraction
  - Batch normalization
- Visualization tools for:
  - Raw data plots
  - Processed data comparison
  - Prediction vs actual curve comparison
- Post-processing using Savitzky-Golay filtering for smooth predictions

## Dependencies
- numpy
- matplotlib
- math
- scikit learn
- tensorflow

## Model Architecture

The model uses a hybrid architecture:
- Input branch 1: LSTM network (8 units) for processing PH curves
- Input branch 2: Dense layers for processing additional features
- Merged outputs through concatenation
- Final dense layer for SS curve prediction

## Results

The model predicts stress-strain curves with the following capabilities:

- Captures the general trend of the stress-strain relationship
- Predicts maximum stress values with reasonable accuracy
- Provides smooth curve predictions after Savitzky-Golay filtering

## Future Work

- Applying the same pipeline to unirradiated samples
- Incorporating additional material features and find out its impact on the predictions
