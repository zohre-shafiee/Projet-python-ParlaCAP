# ParlaCAP — Supervised Classification of Parliamentary Texts

**University group project · Natural Language Processing · Python · Transformer models**

## Overview

This university project explored **supervised classification of parliamentary discourse** using text drawn from the **ParlaCAP** corpus. It combines a language-oriented research question with practical NLP work: preparing labeled text data, experimenting with Transformer-based classifiers and considering model evaluation.

The project is a **team assignment**. This README describes the scope of the coursework and its intended technical workflow; it does not claim that the complete training and evaluation code is present in the public repository.

## Task

The aim was to predict a political text's class from its linguistic content, using labeled examples from parliamentary debates.

A supervised classifier learns patterns from labeled training instances and produces predictions for previously unseen texts. Such predictions can then be compared with reference labels to evaluate performance.

**Important:** Model predictions are statistical outputs. They should not be interpreted as definitive statements about a person's identity or political beliefs.

## General NLP workflow

```text
ParlaCAP parliamentary text corpus
                 |
                 v
       Dataset selection
        and label checking
                 |
                 v
         Text preparation
                 |
                 v
    Train / evaluation split
                 |
                 v
        Transformer model
      experimentation / tuning
                 |
                 v
       Predicted text labels
                 |
                 v
    Comparison with gold labels
```

The precise split, hyperparameters, label mapping and final evaluation scores **cannot be reconstructed from the files currently published here**.

## Methods and concepts explored

- Labeled text datasets for supervised NLP;
- Preparing political discourse for model input;
- Transformer-based text classification, including CamemBERT-family models used in the original course project;
- Model training / fine-tuning experiments;
- Interpreting predictions and evaluating classification performance.

These topics describe the group's university work. The current public notebook is **not** a complete implementation of them.

## Repository contents

```text
.
├── README.md
└── code.ipynb
```

The notebook currently contains a **Google Colab launch link and a Google Drive mount cell only**. It does not yet include the dataset loading, preprocessing, training, inference or evaluation pipeline.

As a result, this repository should be regarded as **documentation of a university project in progress**, not a reproducible model-training package.

## Tools

- Python
- Google Colab / Jupyter Notebook
- Transformer-based NLP models (course experiments)
- ParlaCAP parliamentary corpus

## What would make the project reproducible?

To reproduce the work, this repository would need:

1. The full, reviewed Python notebook or scripts;
2. A documented dataset source, version and access conditions;
3. Exact target label definitions and preprocessing rules;
4. Train / validation / test setup;
5. Model identifiers, training parameters and dependencies;
6. Evaluation metrics and the actual observed results.

These details are intentionally **not invented** in this README.

## Limitations

- The published notebook is incomplete.
- No verified accuracy, F1 score or confusion matrix is available from the files in this repository.
- Training models on political discourse requires careful attention to annotation quality, corpus bias and differences between in-domain and out-of-domain texts.

## Teamwork and contribution

This was completed as **university team coursework**. The project introduced practical experimentation with supervised NLP classification and Transformer models.

This public description does not imply that all original team code was written by one person or that every experimental artefact is currently available online.

## Next steps

The next improvement is to recover and review the final course notebook and supporting report, remove any private tokens or restricted data, and publish an accurately documented reproducible version.
