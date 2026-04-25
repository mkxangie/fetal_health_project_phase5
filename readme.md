# Predicting Fetal Distress: Classifying Fetal Health from Cardiotocography Data

Interpreting Cardiotocography (CTG) traces is critical for detecting fetal distress during pregnancy, but subjective interpretation leads to inconsistency across clinicians. This project uses 2,126 CTG records containing fetal heart rate patterns, variability indicators, and deceleration measurements to classify fetal health as Normal, Suspected, or Abnormal. The final model (Gradient Boosting with SMOTE) achieves a macro F1 of 0.879 and detects 91.4% of Abnormal cases, reducing the most dangerous clinical error -- missed fetal distress -- to just 2 out of 35 cases.

## Business Understanding and Data Understanding

### The problem

Cardiotocography (CTG) is one of the most widely used tools for monitoring fetal health during pregnancy. It measures fetal heart rate and uterine contractions to assess whether a fetus is in distress. However, CTG interpretation is complex and subjective -- studies have shown significant variability between clinicians reading the same trace, which can lead to both over-intervention (unnecessary procedures) and under-intervention (missed distress).

An automated classification system that reliably identifies fetal health status from CTG measurements could assist healthcare providers in making faster, more consistent decisions, reducing inter-observer variability and ensuring at-risk cases are not missed.

**The predictive question:** Using CTG measurements -- fetal heart rate patterns, uterine contractions, decelerations, and variability indicators -- can we classify fetal health status as Normal, Suspected, or Abnormal?

This matters because early and accurate classification enables timely clinical intervention for high-risk pregnancies, potentially reducing adverse outcomes for both mother and baby. In this domain, missing a case of fetal distress (false negative on Abnormal) is far more dangerous than a false alarm, which shapes our entire evaluation strategy.

### The data

The dataset contains 2,126 CTG records with 21 features measuring fetal heart rate patterns, deceleration types, variability indicators, and histogram-derived statistics from the heart rate distribution. The target variable classifies fetal health into three categories: Normal (77.8%), Suspected (13.9%), and Abnormal (8.3%).

Key feature groups include clinical CTG measurements (baseline heart rate, accelerations, fetal movement, uterine contractions, light/severe/prolongued decelerations), variability indicators (abnormal short-term variability, mean short-term variability, percentage of time with abnormal long-term variability, mean long-term variability), and histogram-derived features describing the fetal heart rate distribution.

The severe class imbalance -- with Abnormal cases making up only 8.3% of the data -- is a central challenge addressed in the modelling approach.

### Domain context

Cardiotocography has been a standard tool in obstetric care since the 1960s. While widely used, its interpretation remains subjective, with documented inter-observer and intra-observer variability. Machine learning approaches to CTG classification aim to reduce this variability by providing consistent, data-driven assessments.

Source: Ayres-de-Campos, D., et al. (2015). FIGO consensus guidelines on intrapartum fetal monitoring: Cardiotocography. International Journal of Gynecology & Obstetrics, 131(1), 13-24.

### Data preparation

Correlation analysis revealed redundancy among the histogram features -- histogram_mean, histogram_median, and histogram_min were highly correlated (r > 0.85) with other features and were dropped. The final feature set contained 18 features. No missing values were found in the dataset.

## Modelling and Evaluation

### Models used

We trained and evaluated five models, progressing from simple to complex:

1. **Logistic Regression** -- interpretable baseline
2. **Support Vector Machine (SVM)** -- non-linear decision boundaries
3. **Random Forest** -- ensemble with feature interaction capability
4. **Gradient Boosting** -- sequential error-correction approach
5. **Gradient Boosting + SMOTE** -- addressing class imbalance with synthetic oversampling of the minority classes
6. **Stacked Ensemble + SMOTE** -- combining Random Forest and Gradient Boosting with a Logistic Regression meta-learner to test whether model combination outperforms individual models

All models were evaluated using 5-fold stratified cross-validation and a held-out test set (80/20 split). We used macro F1 as the primary comparison metric, with special attention to recall on the Abnormal class -- the most clinically critical measurement.

### Results

| Model | Test Accuracy | Test Macro F1 | Abnormal Recall |
|-------|:---:|:---:|:---:|
| Logistic Regression | 0.876 | 0.764 | 0.657 |
| SVM | 0.890 | 0.769 | 0.657 |
| Random Forest | 0.925 | 0.847 | 0.800 |
| Gradient Boosting | 0.932 | 0.860 | 0.800 |
| Stacked Ensemble + SMOTE | 0.925 | 0.856 | 0.857 |
| **GB + SMOTE** | **0.934** | **0.879** | **0.914** |
| Majority baseline | 0.778 | -- | 0.000 |

All six models beat the majority class baseline (77.8% accuracy). The final model (GB + SMOTE) achieves the highest scores across all metrics and correctly identifies 32 out of 35 Abnormal cases. A Stacked Ensemble combining Random Forest and Gradient Boosting was also tested -- it performed well (85.7% Abnormal recall) but did not outperform GB + SMOTE, confirming that addressing class imbalance directly was more impactful than combining models for this problem.

### Feature importance

Feature importance was consistent across all three tree-based models. The top predictors were:

1. **Variability indicators** -- abnormal short-term variability, percentage of time with abnormal long-term variability, and mean value of short-term variability dominated across all models, confirming that heart rate variability is the strongest signal for fetal health classification.
2. **Prolongued decelerations** -- consistently ranked in the top five, aligning with clinical knowledge that sustained heart rate drops indicate potential fetal distress.
3. **Accelerations** -- ranked highly, consistent with their role as a reassuring sign in CTG interpretation.
4. **Histogram mode** -- the central tendency of the heart rate distribution carried useful signal even after removing redundant histogram features.

Severe decelerations ranked last despite clinical significance, because they are extremely rare in the dataset (most values are zero).

## Conclusion

The final model demonstrates that automated CTG classification can reliably detect fetal distress. With 91.4% recall on Abnormal cases, the model catches the vast majority of cases requiring intervention while maintaining strong overall accuracy.

**Recommendations:**

- **Healthcare providers**: This model is best suited as a decision-support tool to flag cases for further review, not a replacement for clinical judgment. Clinicians should focus on variability indicators and prolongued decelerations as the most informative CTG measurements.
- **Hospital systems**: Deploying this model as a screening layer could reduce inter-observer variability in CTG interpretation and ensure more consistent triage of high-risk cases.
- **Medical researchers**: The low importance of severe decelerations highlights a data limitation -- future datasets should include more pathological cases to improve learning on rare but critical events.

**Limitations**: The dataset is relatively small (2,126 records), does not capture temporal CTG changes, and may represent a specific population. The model should be validated on external data before clinical deployment. It is intended as a decision-support tool, not a replacement for clinical judgment.

## Repository Navigation

```
fetal-health-classification/
|-- README.md                              <- This file
|-- fetal_health_classification.ipynb      <- Final Jupyter notebook
|-- presentation.pdf                       <- Non-technical presentation
|-- data/
|   |-- fetal_health.csv                   <- Dataset
|-- images/
|   |-- header.png                         <- README header image
|-- requirements.txt                       <- Python package dependencies
```

- **Final notebook**: [fetal_health_classification.ipynb](fetal_health_classification.ipynb)
- **Presentation**: [presentation.pdf](presentation.pdf)

### Reproducing this project

**Environment**: Python 3.10+ (developed using Google Colab)

**Required packages** (install via `pip install -r requirements.txt`):

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
```

**Steps to reproduce**:

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place `fetal_health.csv` in the `data/` directory (or update the file path in the notebook)
4. Run all cells in `fetal_health_classification.ipynb` sequentially

The dataset is included in the repository. It was sourced from a publicly available Cardiotocography dataset used for fetal health classification research.
