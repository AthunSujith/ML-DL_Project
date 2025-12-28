# Amazon Customer Review Analysis using CuDNNLSTM

## Overview
This project implements a sentiment analysis model for Amazon customer reviews using Long Short-Term Memory (LSTM) networks, specifically optimized with Nvidia's CuDNN for faster training on GPUs.

## Project Structure
- `analysis.ipynb`: The main notebook for data analysis and model implementation.
- `req.txt`: List of Python dependencies.
- `train.ft.txt.bz2.zip` & `test.ft.txt.bz2.zip`: Compressed dataset files.
- `glove.twitter.27B.100d.txt.zip`: Pre-trained Glove embeddings.
- `venv/`: Virtual environment directory.

## How to Run
1. Create a virtual environment and install dependencies from `req.txt`.
2. Extract the datasets and embeddings.
3. Run the `analysis.ipynb` notebook.
