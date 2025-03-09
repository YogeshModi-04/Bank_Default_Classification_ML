# 📌 Bank Customer Classification - Machine Learning Project

This repository contains a machine learning project focused on **classifying bank customers** based on their likelihood of defaulting on a loan. It includes **synthetic dataset generation, exploratory data analysis (EDA), and model training** using various classification techniques.

---

## 📜 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Dataset Details](#-dataset-details)
- [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [Machine Learning Model](#-machine-learning-model)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

This project aims to **predict customer loan defaults** using various machine learning techniques. The dataset is **synthetically generated** to simulate real-world banking scenarios and includes customer demographics, financial details, and loan-related attributes.

The pipeline consists of:

1. **Data Generation** - Creating a synthetic dataset with realistic banking features.
2. **Exploratory Data Analysis (EDA)** - Understanding feature distributions, correlations, and patterns.
3. **Preprocessing** - Handling missing values, encoding categorical variables, and feature scaling.
4. **Model Training** - Implementing multiple classification models (e.g., Logistic Regression, Random Forest, Gradient Boosting).
5. **Evaluation** - Comparing model performance using accuracy, precision, recall, F1-score, and AUC-ROC.

---

## 🚀 Features

- 📊 **Exploratory Data Analysis (EDA)**
- 🏦 **Realistic Banking Dataset with 1,000,000 rows**
- 🤖 **Machine Learning Classification Models**
- 🎯 **Hyperparameter Tuning for Performance Optimization**
- 📉 **Handling Class Imbalance with Resampling Techniques**
- 📁 **Export Trained Models for Future Use**

---

## 📂 Dataset Details

The dataset contains **25 features** (numerical & categorical) and a **binary target variable**:

### 🎯 **Target Variable**
- `Default`: Customers who defaulted on their loan.
- `No Default`: Customers who successfully repaid their loan.

### 🔢 **Numerical Features**
- `Age` (18-75)
- `Annual_Income` (~$50,000 mean)
- `Loan_Amount` (~$20,000 mean)
- `Savings_Balance`
- `Years_as_Customer`
- `Credit_Score` (300-850)
- `Debt_to_Income_Ratio`
- `Credit_Utilization`
- `Loan_Term_in_Years`
- `Monthly_Installment`
- **(Total: 13 numerical features)**

### 🔠 **Categorical Features**
- `Gender` (Male, Female, Non-Binary)
- `Marital_Status` (Single, Married, etc.)
- `Employment_Status`
- `Education_Level`
- `Loan_Purpose`
- `Has_Credit_Card` (Yes/No)
- `Is_Homeowner` (Yes/No)
- **(Total: 10 categorical features)**

---

## 🔍 Exploratory Data Analysis (EDA)

The project includes multiple EDA visualizations:

- **📈 Correlation Heatmap:** Identifies relationships between numerical features.
- **📊 Boxplots:** Loan amounts by loan purpose.
- **📉 Bar Charts:** Target distribution (default vs. no default).
- **📊 Histograms:** Distribution of numerical features.

---

## 🤖 Machine Learning Model

The classification models implemented include:

- **Logistic Regression**
- **Random Forest**
- **Gradient Boosting (XGBoost)**
- **Support Vector Machine (SVM)**
- **Neural Network (MLPClassifier)**

📌 **Performance Metrics:**
- Accuracy
- Precision & Recall
- F1-Score
- AUC-ROC Curve

---

## 💾 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/bank-customer-classification.git
   cd bank-customer-classification
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

---

## ⚙️ Usage

1. **Run `Bank_customer_classification.ipynb` in Jupyter Notebook.**
2. **Explore the dataset & EDA visualizations.**
3. **Train models and evaluate performance.**
4. **Fine-tune hyperparameters for better accuracy.**
5. **Save and export trained models.**

---

## 📁 Project Structure

```bash
bank-customer-classification/
├── Bank_customer_classification.ipynb  # Jupyter Notebook (Main Project)                  
├── README.md                            # Documentation
└── results/                             # Model evaluation results
```

---

## 📊 Results

- The best-performing model is **[INSERT BEST MODEL]** with an accuracy of **[INSERT VALUE]%**.
- The **AUC-ROC Curve** shows that the model effectively separates loan defaulters from non-defaulters.

Sample Confusion Matrix:
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, model.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d')
```

---

## 🤝 Contributing

🔹 Feel free to **open an issue** for bug reports, feature requests, or general improvements.  
🔹 You can also **fork the repository** and submit a **pull request** with your enhancements.

---

## 📜 License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---

## 📩 Contact

For questions or collaboration opportunities, reach out via:

📧 Email: [skmodiyogesh@gmail.com]  
🔗 GitHub: [YogeshModi-04](https://github.com/YogeshModi-04)

---

🔥 **Happy Coding!** 🚀
