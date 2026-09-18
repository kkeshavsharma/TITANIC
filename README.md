# Titanic Survival Prediction: From Basic ML to Feature Engineering

A progressive machine learning project using the **Kaggle Titanic dataset** to learn and improve a classification workflow step by step.

This repository contains **three versions of the same project**, developed sequentially. Instead of replacing the earlier approach, I kept each version to document how my understanding of data preprocessing, feature engineering, model selection, evaluation, and machine learning experimentation improved over time.

## Dataset

This project uses the **Titanic - Machine Learning from Disaster** dataset from Kaggle.

Dataset: https://www.kaggle.com/competitions/titanic/data

The objective is to predict whether a passenger survived based on information such as:

* Passenger class
* Sex
* Age
* Number of siblings/spouses
* Number of parents/children
* Fare
* Embarkation point
* Ticket and cabin-related information

---

# Project Evolution

## Version 0 - Basic Classification

**Model:** Random Forest Classifier

The first version focuses on understanding the basic machine learning pipeline:

```text
Load Data
   ↓
Basic Data Cleaning
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
Random Forest
   ↓
Prediction
   ↓
Evaluation
```

### What I learned

* Loading and inspecting a real-world dataset using Pandas
* Handling missing values
* Converting categorical variables into numerical values
* Separating features (`X`) and target (`y`)
* Performing a train/test split
* Training a Random Forest classifier
* Understanding feature importance
* Making class predictions and probability predictions
* Evaluating a classification model using:

  * Accuracy
  * Confusion Matrix
  * Classification Report

### Initial preprocessing

The first version deliberately used a relatively simple feature set:

```text
Pclass
Sex
Age
SibSp
Parch
Fare
Embarked
```

Columns such as `Name`, `Cabin`, `Ticket`, and `PassengerId` were initially removed.

This gave me a foundation for understanding how a complete ML classification workflow works before moving toward more advanced feature engineering.

---

# Version 2 - Feature Engineering + XGBoost

**Model:** XGBoost Classifier

After understanding the basic workflow, I moved beyond simply dropping complicated columns.

The goal of Version 1 was to investigate whether information hidden inside `Ticket`, `Cabin`, and passenger characteristics could be converted into useful machine learning features.

### New features

#### Ticket Prefix

Extracted alphabetic information from the ticket and converted it into numerical categories.

#### Duplicate Ticket Count

Calculated how many passengers shared the same ticket:

```python
dup_ticket = df.groupby("Ticket").size()
df["DupTicket"] = df["Ticket"].map(dup_ticket)
```

This attempts to capture information about passengers traveling together.

#### Ticket Survival Rate

Created a ticket-level survival feature based on children and female passengers:

```python
Tick_surv = df[boy_or_female].groupby("Ticket").Survived.mean()
df["TicketSurv"] = df["Ticket"].map(Tick_surv)
```

This was my first attempt to extract **group-level information** rather than treating every passenger as completely independent.

#### Cabin Deck

Instead of using the entire cabin value, I extracted the first character to represent the cabin deck.

### Model improvement

Version 0 used:

```text
Random Forest
```

Version 1 moved to:

```text
XGBoost
```

and introduced hyperparameters such as:

* `max_depth`
* `min_child_weight`
* `learning_rate`
* `n_estimators`
* `subsample`
* `colsample_bytree`
* `scale_pos_weight`

### Additional learning

Version 1 also introduced SQLite:

```python
conn = sqlite3.connect("titanic.db")
df.to_sql("titanic", conn, if_exists="replace", index=False)
```

This gave me exposure to storing and querying dataset information using SQL rather than working exclusively with Pandas.

### What I learned

* Feature engineering from raw columns
* Group-based features
* Extracting information from strings
* Understanding how domain information can become ML features
* XGBoost classification
* Handling class imbalance with `scale_pos_weight`
* Basic hyperparameter tuning
* Using SQLite with Python
* Thinking about data leakage instead of blindly fitting a model

---

# Version 3 - EDA + More Systematic Preprocessing

**Model:** XGBoost Classifier

Version 3 focused more heavily on understanding the data **before** training the model.

Instead of immediately cleaning and fitting a model, I added an exploratory data analysis stage.

## Exploratory Data Analysis

I examined:

* Survival distribution
* Passenger class distribution
* Sex distribution
* Sibling/spouse counts
* Parent/child counts
* Fare distribution
* Age distribution
* Embarkation distribution

Visualizations were created using Matplotlib and Seaborn.

This helped me understand the structure and distributions of the dataset before feature engineering.

---

## Feature Engineering

Version 3 introduced additional transformations.

### Passenger Title

Titles were extracted from passenger names:

```python
df["Title"] = df["Name"].str.extract(r",\s*([A-Za-z]+)\.")
```

This converts information such as passenger titles into a potentially useful predictive feature.

### Ticket Prefix

Instead of keeping the complete ticket string, I extracted its textual component.

### Cabin Information

Cabin information was transformed to capture the categorical cabin information in numerical form.

### Encoding

`LabelEncoder` was introduced to convert categorical values into numerical representations.

---

# Missing Value Handling

A major change in Version 3 was experimenting with **Iterative Imputation**.

Instead of simply replacing missing values with a single statistic such as the mean or median, I used:

```python
from sklearn.impute import IterativeImputer
```

The imputer was fitted on the training data and then used to transform the data.

The intention was to learn how more advanced missing-value strategies work and understand the importance of separating training and testing data during preprocessing.

---

# Model

Version 3 continued using XGBoost, but experimented with a different configuration:
```text
n_estimators = 1000
learning_rate = 0.01
max_depth = 3
subsample = 0.3
gamma = 0.9
min_child_weight = 3
reg_alpha = 3
reg_lambda = 2
```

This version gave me more exposure to regularization and controlling model complexity rather than simply increasing model size.

---

# Evaluation

The models were evaluated using:

* Accuracy
* Confusion Matrix
* Classification Report
* Training performance
* Feature importance

The confusion matrix was also visualized to understand the types of classification errors being made.

---

# What Changed Across the Versions?

| Area                           | Version 0         | Version 1         | Version 3              |
| ------------------------------ | ----------------- | ----------------- | ---------------------- |
| Model                          | Random Forest     | XGBoost           | XGBoost                |
| Basic preprocessing            | Yes               | Yes               | Yes                    |
| Exploratory Data Analysis      | Limited           | Limited           | Yes                    |
| Feature engineering            | Basic             | Extensive         | More structured        |
| Ticket features                | No                | Yes               | Yes                    |
| Cabin features                 | Dropped           | Deck              | Encoded                |
| Passenger titles               | No                | No                | Yes                    |
| Group information              | No                | Ticket survival   | Expanded preprocessing |
| Missing values                 | Simple imputation | Simple imputation | Iterative Imputation   |
| Class weighting                | No                | Yes               | Yes                    |
| Feature importance             | Yes               | Yes               | Yes                    |
| Confusion matrix               | Yes               | Yes               | Yes                    |
| SQLite                         | No                | Yes               | No                     |
| Hyperparameter experimentation | Basic             | Yes               | Yes                    |

---

# Learning Progression

The main purpose of keeping all three versions is to show the progression in my understanding of machine learning.

### Version 0

**"Can I build a complete classification model?"**

I learned the fundamental ML pipeline:

```text
Data → Cleaning → Features → Train/Test Split → Model → Prediction → Evaluation
```

### Version 1

**"Can I extract more information from the raw data?"**

I started looking at the dataset from a feature-engineering perspective instead of simply deleting difficult columns.

I learned that raw data can contain useful information that is not immediately represented as a numerical feature.

### Version 3

**"Can I build the pipeline more systematically?"**

I added exploratory data analysis, additional feature engineering, more advanced imputation, encoding, and further XGBoost experimentation.

The focus shifted from simply getting a model to run toward understanding **why preprocessing and feature construction affect the model**.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* SQLite

---

# Repository Structure

```text
Titanic-Survival-Prediction/
│
├── version1.ipynb
├── version2.ipynb
├── version3.ipynb
├── train.csv
├── titanic.db
└── README.md
```
---

# Important Note About the Experiments

The three versions are **learning iterations**, not three independently optimized competition submissions.

The purpose was to progressively understand:

```text
Basic ML
   ↓
Feature Engineering
   ↓
Advanced Preprocessing
   ↓
Model Experimentation
```

Therefore, differences in accuracy between versions should not be interpreted as a strict benchmark of model quality. Changes in the train/test split, preprocessing, feature construction, and model configuration can all affect the results.

---

# Key Takeaways

Through this project I practiced:

* Real-world data cleaning
* Exploratory data analysis
* Categorical encoding
* Missing-value handling
* Feature engineering
* Group-based features
* Classification
* Random Forest
* XGBoost
* Hyperparameter experimentation
* Model evaluation
* Confusion matrix analysis
* Feature importance
* SQL/SQLite integration
* Thinking about data leakage

Most importantly, the project documents the transition from **using a machine learning library to understanding the decisions around the model**.

---

## Future Improvements

GOOD THINGS DO END

---

## Project Goal

This project is part of my ongoing effort to move from simply applying ML algorithms toward understanding the complete process of **data analysis, preprocessing, feature engineering, model training, and evaluation**.
