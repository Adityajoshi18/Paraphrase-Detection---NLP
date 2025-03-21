# Paraphrase-Detection---NLP

Developed a Siamese Deep Network called MaLSTM (Manhattan LSTM) to detect similarity between two sentences.

## Overview

This project aims to build a machine learning model for paraphrase detection. Paraphrase detection is the task of determining whether two given sentences convey the same meaning. The model is trained and evaluated on a dataset consisting of pairs of sentences labeled as paraphrases or non-paraphrases.

## Dataset

The project utilizes the following datasets:

- train.csv: Contains labeled sentence pairs for training.
- test.csv: Contains sentence pairs for testing the model's performance.
- sample_submission.csv: A sample format for the submission file.

## Project Files

- Paraphrase Detection.ipynb: A Jupyter Notebook containing the implementation of data preprocessing, model training, and evaluation.
- train.csv: Training dataset with labeled sentence pairs.
- test.csv: Testing dataset for model evaluation.
- sample_submission.csv: Template for submission.
- README.md: This file containing project documentation.

## Dependencies

To run this project, install the following Python libraries:

```bash
   pip install pandas numpy scikit-learn tensorflow transformers
```

## Steps to Run

1. Load the dataset (train.csv and test.csv).

2. Preprocess the text data (tokenization, stopword removal, etc.).

3. Train the MaLSTM model for sentence similarity detection.

4. Evaluate the model performance using appropriate metrics (e.g., accuracy, F1-score).

5. Generate predictions and format them according to sample_submission.csv.

## Model

The project utilizes the MaLSTM (Manhattan LSTM), a Siamese Deep Network that computes the similarity between two sentences based on their representations.

## Evaluation Metrics

- Accuracy
- Precision, Recall, and F1-score

## How to Contribute

- Improve preprocessing techniques
- Experiment with different model architectures
- Fine-tune transformer models for better accuracy
- Optimize hyperparameters for better performance

## Contact

For any questions or contributions, feel free to reach out!
