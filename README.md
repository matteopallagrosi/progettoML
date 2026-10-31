# Sentiment Analysis of IMDb and Yelp Reviews

## Overview

This project explores **Sentiment Analysis**, a Natural Language Processing (NLP) task aimed at automatically identifying the sentiment expressed in textual data.

The objective is to develop and compare different Machine Learning and Deep Learning approaches for binary sentiment classification, distinguishing between positive and negative reviews.

The experiments are conducted on two domains, IMDb (movie reviews) and Yelp (restaurant reviews), considering both individual datasets and their combination. The project investigates the effectiveness of traditional machine learning algorithms, neural network architectures and Transformer-based language models.

## Dataset

The project uses two subsets of the [Sentiment Labelled Sentences Dataset](https://archive.ics.uci.edu/dataset/331/sentiment%2Blabelled%2Bsentences) from the UCI Machine Learning Repository:

- **IMDb:** movie review sentences.
- **Yelp:** restaurant review sentences.

Each subset contains 1,000 labeled sentences, equally distributed between positive and negative sentiment (500 samples per class).

Sentiment labels are binary:
- `1`: Positive
- `0`: Negative

The experiments consider three evaluation settings:
- IMDb dataset only.
- Yelp dataset only.
- Combined IMDb and Yelp datasets.

## Models

Several approaches are implemented and evaluated to compare different strategies for text representation and sentiment classification.

### Traditional Machine Learning

- **Logistic Regression:** classification using TF-IDF feature extraction.
- **Logistic Regression with TF-IDF tuning:** hyperparameter optimization to investigate the impact of different text vectorization configurations.
- **Naive Bayes:** probabilistic classification approach for text categorization.

### Deep Learning

- **LSTM:** recurrent neural network designed to capture sequential dependencies in text.
- **LSTM with Pre-trained Embeddings:** LSTM architecture enhanced with pre-trained word embeddings to leverage semantic information.
- **Transformer:** attention-based architecture for modeling relationships between words in a sentence.
- **BERT Fine-tuning:** adaptation of a pre-trained BERT language model to the binary sentiment classification task.

## Evaluation and Results

Model performance is evaluated using two classification metrics:

- **Accuracy:** proportion of correctly classified samples.
- **F1-score:** harmonic mean of precision and recall.

The following table summarizes the accuracy achieved by each model across the three evaluation settings, together with its F1-score on the combined dataset.

| Model | IMDb Accuracy | Yelp Accuracy | Combined Accuracy | Combined F1 |
|:--|--:|--:|--:|--:|
| Logistic Regression | 82.7% | 75.3% | 81.0% | 81.0% |
| LR + TF-IDF Tuning | 82.0% | 76.7% | 81.0% | 81.0% |
| Naive Bayes | 80.0% | 76.0% | 80.0% | 80.0% |
| LSTM | 82.7% | 76.7% | 77.0% | 76.8% |
| LSTM + Pre-trained Embeddings | 85.3% | 80.7% | 83.3% | 83.3% |
| Transformer | 83.3% | 82.0% | 84.0% | 84.0% |
| BERT Fine-tuning | **90.0%** | **83.3%** | **87.0%** | **87.0%** |

BERT fine-tuning achieves the best overall performance, with 87% accuracy on the combined dataset and 90% accuracy on IMDb.

The results highlight the effectiveness of pre-trained language models in capturing contextual information and semantic relationships, outperforming the traditional machine learning and neural network approaches considered in these experiments.

Detailed results are available in `accuracy.csv` and `f1.csv`.

## Repository Structure

```text
progettoML/
│
├── project_sentiment_analysis.ipynb  # Main project notebook
├── accuracy.csv                      # Accuracy results
├── f1.csv                            # F1-score results
├── presentazioneML.pdf               # Project presentation
└── README.md                         # Project documentation
```

## Getting Started

### Dataset Preparation

Download the Sentiment Labelled Sentences dataset from the UCI Machine Learning Repository.

Extract the following files:
- `imdb_labelled.txt`
- `yelp_labelled.txt`

Place them in the locations expected by the notebook, adjusting the dataset paths if necessary.

### Running the Notebook

Open the main Jupyter Notebook:

```bash
jupyter notebook project_sentiment_analysis.ipynb
```

Install the Python dependencies imported by the notebook before executing the cells.

## References

- Kotzias, D. (2015). *Sentiment Labelled Sentences*. UCI Machine Learning Repository. DOI: `10.24432/C57604`.
