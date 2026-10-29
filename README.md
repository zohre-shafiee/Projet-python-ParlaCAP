# ParlaCAP — Classifying French Parliamentary Speeches with Transformers

**University NLP project · Two-person team · Université Grenoble Alpes**

**Authors: Zohré Shafiee and Camille Benoit-Guyod**

## Project overview

We trained and compared Transformer-based classifiers to **predict the political party associated with a French parliamentary speech**, using the `text` and `speaker_party` fields from the ParlaCAP corpus. This is a supervised **multiclass text classification** task, not an attempt to infer an individual's personal political beliefs.

The project explores the effects of dataset cleaning, class imbalance, model choice, sequence length and training duration. Five experiments were conducted with **DistilCamemBERT** and **CamemBERT-base**.

The source scripts and recorded results are included here. **The reported metrics come from our university runs; they have not been independently rerun for this public repository.**

## Dataset and preparation

The dataset contains French parliamentary speeches dated **2015–2022**, provided as a TSV file with speech text, speaker and party metadata.

**Dataset source:** https://data.crossda.hr/file.xhtml?persistentId=doi:10.23669/1ZTELP/TRB27P&version=1.0

The original dataset is **not distributed in this repository**.

### Processing pipeline

1. Read `parlacap.tsv` and select speech text and the speaker's party.
2. Remove missing or `-` party labels.
3. Exclude speeches with **15 words or fewer**.
4. In the later experiments, group parties representing **less than 5%** of the retained data into `AUTRE` ("Other").
5. Convert party names into integer class labels.
6. In experiments 2–5, perform an **80/20 stratified train/evaluation split** using seed 42.
7. Tokenize with the pretrained model's tokenizer, truncate to the configured length and apply dynamic padding.
8. Fine-tune a sequence classification model and calculate **accuracy** and **macro-averaged F1**.

```text
ParlaCAP TSV
     |
     v
Clean speech texts and party labels
     |
     v
Group low-frequency parties (later versions)
     |
     v
Encode labels + 80/20 split
     |
     v
DistilCamemBERT / CamemBERT tokenizer
     |
     v
Fine-tune Transformer classifier
     |
     v
Accuracy + macro-F1 + class-level report
```

### Data overview

| Measure | Before cleaning | After cleaning |
|---|---:|---:|
| Speeches | 714,439 | 262,257 |
| Share retained | — | 36.7% |
| Median length | 13 words | 72 words |
| Party categories | 18 | 7 |

After cleaning, the seven label categories were:

| Party | Share |
|---|---:|
| LAREM | 32.49% |
| LR | 23.37% |
| AUTRE | 16.08% |
| FI | 9.63% |
| SOC | 6.85% |
| GDR | 6.13% |
| MODEM | 5.45% |

The original data also included entries with missing or unspecified party labels (approximately 15.15% labeled `-`). A rough token-count estimate in the exploratory analysis suggested **5.01%** of cleaned speeches would exceed 512 tokens. This was an **estimate based on word counts**, not a direct measurement with the final tokenizer.

## Model architectures and training

All experiments used Hugging Face `transformers`, `datasets`, `evaluate` and PyTorch-based training.

- **DistilCamemBERT:** `cmarkea/distilcamembert-base` (experiments 1–2).
- **CamemBERT-base:** `camembert-base` (experiments 3–5).
- Learning rate: `2e-5`; training batch size: `8`; evaluation batch size: `8`; weight decay: `0.01`.
- Most configurations used **three epochs**; experiment 5 used **four**.
- Later configurations selected the best checkpoint using **macro-F1**.

Tokenization produces input IDs and attention masks; `DataCollatorWithPadding` pads examples in each batch. Model logits are mapped to the class with the highest score, and predictions are compared with reference labels.

### Five experiments

| Experiment | Script | Model | Epochs | Accuracy | Macro-F1 | Evaluation loss |
|---|---|---|---:|---:|---:|---:|
| 1 | `projetpython.py` | DistilCamemBERT | 3 | 54.06% | 29.16% | 1.4264 |
| 2 | `projetpython_v2.py` | DistilCamemBERT | 3 | 53.68% | 43.52% | 1.2625 |
| 3 | `projetpython_v3.py` | CamemBERT-base | 3 | 56.41% | 47.25% | 1.2364 |
| 4 | `projetpython_v4.py` | CamemBERT-base, weighted loss | 3 | 51.14% | 47.35% | 1.4133 |
| 5 | `projetpython_v5.py` | CamemBERT-base | 4 | **56.69%** | **48.29%** | 1.3327 |

These results were taken from the team's experiment notes and are **not new measurements**.

#### Experiment 1 — Initial baseline

The first run fine-tuned DistilCamemBERT for three epochs. The original experiment notes describe a **2% threshold** for grouping rare parties; the recovered first-version script does **not** implement that grouping. This discrepancy should be kept in mind when attempting to reproduce the historical run.

The initial macro-F1 of **29.16%** was considerably lower than its **54.06% accuracy**, suggesting uneven performance across party classes.

#### Experiment 2 — Better class handling

We moved the rare-party grouping threshold to **5%**, created an `AUTRE` category, used a **stratified split**, selected the best checkpoint by macro-F1 and limited inputs to **256 tokens**.

Macro-F1 increased to **43.52%**, while accuracy remained close to the first result. This suggests that balancing the label definition improved average class-level performance without a similar improvement in overall accuracy.

#### Experiment 3 — CamemBERT and longer inputs

We switched to **CamemBERT-base** and increased the maximum length to **512 tokens**, training for three epochs.

Accuracy reached **56.41%** and macro-F1 **47.25%**.

The detailed `rapport_v3.txt` report shows marked class differences:

| Party | Precision | Recall | F1 |
|---|---:|---:|---:|
| AUTRE | 0.47 | 0.40 | 0.44 |
| FI | 0.72 | 0.52 | 0.60 |
| GDR | 0.50 | 0.48 | 0.49 |
| LAREM | 0.61 | 0.80 | 0.69 |
| LR | 0.52 | 0.58 | 0.55 |
| MODEM | 0.55 | 0.14 | 0.22 |
| SOC | 0.44 | 0.24 | 0.31 |

The report contains **52,452 evaluation examples**. LAREM had relatively high recall, while MODEM and SOC were harder to identify. Topic overlap, class imbalance and speeches without strongly discriminative linguistic clues are possible explanations, not proven causal findings.

#### Experiment 4 — Class-weighted loss

We implemented a custom `WeightedTrainer` using PyTorch `CrossEntropyLoss(weight=...)` to give less frequent classes greater weight.

Macro-F1 was **47.35%** and accuracy **51.14%**. This experiment did not provide a clear overall improvement over experiment 3 and was not chosen as the preferred setting.

#### Experiment 5 — Four training epochs

We returned to the unweighted CamemBERT-base configuration with 512-token inputs and increased training from **three to four epochs**.

This produced the best **reported overall combination**: **56.69% accuracy**, **48.29% macro-F1** and **1.3327 evaluation loss**.

The F1 gain over experiment 3 was modest; improvements should not be overstated.

## Files

```text
.
├── README.md
├── clean_data.py         # Dataset exploration and cleaning statistics
├── projetpython.py       # First DistilCamemBERT configuration
├── projetpython_v2.py    # Class grouping and stratified split
├── projetpython_v3.py    # CamemBERT-base, 512 tokens
├── projetpython_v4.py    # Custom class-weighted training loss
├── projetpython_v5.py    # CamemBERT-base, four epochs
├── rapport_v3.txt        # Experiment 3 per-class classification report
└── code.ipynb            # Earlier Google Colab starter notebook
```

The Python scripts are the recovered experimentation code. The older `code.ipynb` contains only initial Colab setup and does not replace them.

## Installation and running an experiment

Install Python and the relevant packages:

```bash
python -m pip install transformers datasets evaluate accelerate numpy scikit-learn torch sentencepiece protobuf pandas
```

Download the source dataset and place `parlacap.tsv` in the working directory. Then run one of the experiment scripts:

```bash
python projetpython_v5.py
```

To explore dataset statistics:

```bash
python clean_data.py
```

**Note:** `clean_data.py` originated in Google Colab and contains a Google Drive file path; adjust it before running locally. The Transformer experiments require downloading pretrained models, access to the corpus and sufficient computation resources. Package versions and exact runtime environments were not pinned in the recovered files.

### Main functions

- `group_small_parties`: maps classes below the frequency threshold to `AUTRE`.
- `preprocess_function`: tokenizes the input text and applies length limits.
- `compute_metrics`: computes accuracy and macro-F1 from predictions and references.
- `WeightedTrainer.compute_loss`: in experiment 4, substitutes a class-weighted cross-entropy loss.

## Interpretation and methodological limitations

**Evaluation leakage:** Experiments 2–5 use the split named `test` as the `eval_dataset` during training and for best-checkpoint selection (`load_best_model_at_end=True`). The reported scores are therefore **not a completely independent final test evaluation**. A more rigorous follow-up would use distinct training, validation and held-out test sets.

Other limitations include:

- The cleaned corpus still contains **substantial class imbalance**.
- `AUTRE` contains politically heterogeneous parties.
- Party membership is only a **dataset label**, not a complete characterization of speech content.
- Long documents are **truncated**, potentially losing information.
- Accuracy may hide poor minority-class recall; macro-F1 and per-class reporting help reveal it.
- The repository contains experimental scripts rather than a fully packaged, version-pinned and independently reproduced research pipeline.

An encoding-related issue was reported while generating the detailed classification report. The team resolved it by running the script with `python3` in the project's virtual environment.

For a Transformer of sequence length `n`, self-attention has an approximate **O(n²)** cost in sequence length. Across `N` examples, a simplified attention-related cost is **O(N × n²)**, omitting other model operations and factors.

## Team and credits

**This project was carried out by two students: Zohré Shafiee and Camille Benoit-Guyod**, as part of the university NLP / Python coursework at Université Grenoble Alpes.

The code, experimentation and results are collaborative academic work. This repository does not claim sole authorship of the project by either student.
