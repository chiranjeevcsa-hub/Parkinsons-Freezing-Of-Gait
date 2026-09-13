# Parkinson's Freezing of Gait Detection

An end-to-end machine-learning pipeline for detecting possible Freezing of Gait (FoG) episodes from wearable accelerometer data.

Freezing of Gait is a brief episode in which a person with Parkinson's disease may feel unable to start or continue walking, as if their feet are temporarily stuck to the ground.

This project uses the Daphnet Freezing of Gait dataset. It focuses on the complete workflow required to turn raw sensor recordings into interpretable, timestamped FoG predictions.

This is an educational machine-learning prototype. It is not a clinical diagnostic system.

## Project objectives

The project was designed to answer the following questions:

1. Can raw accelerometer recordings be loaded and validated reliably?
2. What do the recordings, labels, participants, and sampling intervals look like?
3. Can continuous sensor streams be converted into useful time windows?
4. Can simple statistical features distinguish FoG from non-FoG movement?
5. Which classical machine-learning model performs best under a participant-level evaluation?
6. Can predictions be mapped back to the time intervals in which they occurred?

## Dataset

The Daphnet recordings are not standard CSV files. The raw data consists of multiple text files with:

- Values separated by variable whitespace
- No column headings
- Timestamped accelerometer measurements
- Sensors positioned at the ankle, upper leg, and trunk
- Annotation labels for movement state

The recordings use a sampling frequency of approximately 64 Hz. The project contains 17 recordings and approximately 1.9 million raw sensor readings.

The raw files use labels with the following meaning:

| Label | Meaning |
|---:|---|
| 0 | Outside the experiment |
| 1 | Experiment, no FoG |
| 2 | Experiment, FoG |

For modelling, labels 1 and 2 are converted into a binary target:

| Target | Meaning |
|---:|---|
| 0 | No FoG |
| 1 | FoG |

The raw dataset and generated Parquet files are intentionally kept local. They are excluded from Git because of their size and because the repository should contain reproducible code rather than a large copy of the dataset.

## Why the data pipeline matters

The project does not begin with `pd.read_csv("data.csv")`. The loader first reads whitespace-separated files without headers, assigns the expected column names, validates the structure and labels, extracts participant and recording identifiers, and combines the recordings into a local Parquet dataset.

This step is important because errors in file interpretation, labels, recording boundaries, or participant identity would affect every later modelling result.

## Pipeline

The complete workflow is:

```text
Raw sensor files
        ↓
Validated data loader
        ↓
Integrated local Parquet dataset
        ↓
EDA and data-quality checks
        ↓
Binary labels and chronological ordering
        ↓
Two-second sensor windows
        ↓
Statistical window features
        ↓
Participant-level train/test split
        ↓
Four classical ML models
        ↓
Model comparison
        ↓
Timestamped FoG predictions
```

### Windowing

The recordings are divided into non-overlapping two-second windows. At 64 Hz, each window contains approximately 128 consecutive readings.

Windows are created separately within each participant and recording. A window is never allowed to combine readings from different recordings or participants. Its target label is assigned from the majority label within that window.

### Feature engineering

For each sensor channel in each window, the following features are calculated:

- Mean
- Standard deviation
- Minimum
- Maximum
- Range

These features convert a sequence of raw readings into one compact row that can be used by classical machine-learning models.

### Data splitting

The data is split by participant rather than by individual rows. This prevents readings from the same participant appearing in both the training and testing sets.

This is important for sensor data because a random row-level split could allow the model to learn person-specific movement patterns and produce an overly optimistic result.

## Models evaluated

The following models were trained and compared:

1. Logistic Regression
2. Random Forest
3. Support Vector Machine with an RBF kernel
4. Gradient Boosting

Class balancing was used where appropriate because FoG windows are less frequent than non-FoG windows.

## Results

The models were evaluated on a participant-level test split.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Gradient Boosting | 0.884 | 0.580 | 0.640 | 0.608 |
| Support Vector Machine | 0.869 | 0.527 | 0.658 | 0.585 |
| Logistic Regression | 0.879 | 0.573 | 0.545 | 0.559 |
| Random Forest | 0.862 | 0.667 | 0.036 | 0.068 |

Gradient Boosting was selected as the final model because it achieved the best F1-score and provided the best balance between recall and precision. The SVM achieved slightly higher recall, but with lower precision and F1-score.

The Random Forest result illustrates why accuracy alone is not sufficient. Although its accuracy was 0.862, its recall was only 0.036, meaning that it detected very few of the actual FoG windows.

## Timestamp interpretation

The model does not directly perform exact timestamp regression. It predicts whether a two-second window contains FoG.

Each window retains its original start and end timestamps. After prediction, the predicted label is attached to those timestamps. The output therefore identifies intervals such as:

```text
Recording: R01
Window: 20.0–22.0 seconds
Prediction: FoG
```

The timestamp represents the predicted window, not the exact physiological onset of freezing.

## Repository structure

```text
Parkinsons_Fog_Detection/
├── data/
│   ├── raw/                         # Local raw dataset, ignored by Git
│   └── processed/                   # Local Parquet tables, ignored by Git
├── models/                          # Saved model artefacts
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_logistic_regression.ipynb
│   ├── 05_random_forest.ipynb
│   ├── 06_svm.ipynb
│   ├── 07_gradient_boosting.ipynb
│   ├── 08_model_comparison.ipynb
│   └── 09_final_pipeline.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   └── build_dataset.py
├── tests/
│   ├── test_data_loader.py
│   └── test_build_dataset.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Limitations

- The evaluation uses one participant-level train/test split.
- The dataset is limited to the available participants and recording conditions.
- A two-second window gives an interval-level prediction, not an exact clinical onset time.
- The reported results should not be interpreted as clinical performance.
- External validation on new participants and recording conditions would be required before any real-world use.

## Future improvements


Possible extensions include participant-level cross-validation, temporal smoothing of consecutive predictions, richer frequency-domain features, calibration of prediction thresholds, and evaluation on an independent dataset.


## Repository

[Parkinsons_Fog_Detection on GitHub](https://github.com/chiranjeevcsa-hub/Parkinsons_Fog_Detection)
