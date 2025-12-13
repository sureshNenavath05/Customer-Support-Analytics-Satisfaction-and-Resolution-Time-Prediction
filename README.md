# 🛠 Customer Support Analytics: Satisfaction & Resolution Time Prediction

## 📘 Project Overview

Customer satisfaction is a key driver of business success and retention.  
With digital platforms generating massive amounts of support interactions, analyzing and predicting satisfaction levels helps companies enhance service quality, customer experience, and brand loyalty.

This project leverages **machine learning** and **natural language processing (NLP)** to analyze customer support ticket data and predict satisfaction levels.  
The goal is to identify factors influencing satisfaction and develop predictive models that flag dissatisfied customers early.

**Key Questions:**
- What factors in customer support interactions most influence satisfaction ratings?  
- How can predictive models help identify dissatisfied customers early?  
- What insights can help improve service performance and customer experience?

---

## 📊 Dataset Description

**Records:** 8,469 | **Attributes:** 17  
**Target Variable:** `Customer_Satisfaction_Rating` (1–5 scale)  

The dataset contains detailed information about customer support tickets, including demographics, product details, ticket characteristics, communication channels, and response times.

| Feature Category | Example Columns | Description |
| ---------------- | ---------------- | ------------ |
| Customer Info | `Age`, `Gender`, `Email` | Demographic details of customers |
| Ticket Details | `Ticket Type`, `Priority`, `Channel` | Nature and urgency of the issue |
| Time Metrics | `First Response Time`, `Time to Resolution` | Efficiency of service |
| Text Data | `Subject`, `Description`, `Resolution` | Used for NLP sentiment analysis |
| Output | `Customer Satisfaction Rating` | Rating from 1 (low) to 5 (high) |

---

## 🧹 Data Preprocessing

| Step | Description |
| ---- | ------------ |
| **Duplicate Check** | No duplicate records found |
| **Missing Values** | Found in `Resolution`, `First Response Time`, `Time to Resolution`, and `Satisfaction Rating` |
| **Feature Cleaning** | Removed irrelevant columns (`Ticket ID`, `Customer Name`, `Email`, `Status`) |
| **Datetime Conversion** | Extracted `purchase_month`, `purchase_year`, and `Is_Weekend_Purchase` |
| **Feature Engineering** | Derived efficiency ratios (`Resolution_vs_Response`, `Total_vs_Resolution`) |
| **Sentiment Analysis** | Extracted sentiment from `Description` and `Resolution` using **TextBlob** and **VADER** |
| **Keyword Indicators** | Binary flags for ~50 keywords (`error`, `refund`, `urgent`, `payment`, etc.) |
| **Categorical Encoding** | One-hot encoding for categorical variables |
| **Target Binarization** | Converted ratings → `Satisfied (1)` if >3, else `Unsatisfied (0)` |

Final dataset included **115 engineered features** combining numeric, categorical, and sentiment-based variables.

---

## 🔍 Exploratory Data Analysis (EDA)

**Key Insights:**
- Satisfaction ratings are evenly distributed across levels 1–5 (~20% each).  
- Most customers are between **30–60 years old** — working-age adults.  
- Ticket priorities are balanced: Low (23%), Medium (25%), High (25%), Critical (26%).  
- Major communication channels: **Email, Chat, Social Media, Phone** (nearly equal usage).  
- Top ticket types: **Refund Requests**, **Technical Issues**, and **Billing Inquiries**.  
- Satisfaction is lower for issues like **Network Problems** and **Delivery Delays**.  
- Faster **response** and **resolution times** correlate with higher satisfaction ratings.  

📊 **Observation:**  
Response efficiency and resolution quality significantly impact customer satisfaction, while long delays often correspond with low ratings.
<img width="1227" height="723" alt="image" src="https://github.com/user-attachments/assets/14df44fb-93f0-4264-8d2e-89e7364e0ab2" />
<img width="1237" height="700" alt="image" src="https://github.com/user-attachments/assets/f5f1d2c1-2bab-4987-b91d-f79e270c8d8b" />
<img width="1251" height="702" alt="image" src="https://github.com/user-attachments/assets/1b9e1ef1-b3ea-4fb6-8475-9fbb334c9a53" />
<img width="1241" height="702" alt="image" src="https://github.com/user-attachments/assets/f56371d8-f84f-46cd-972d-2d510feb0e00" />
<img width="1242" height="695" alt="image" src="https://github.com/user-attachments/assets/8d6e0622-5a8d-405f-90f0-d9bb0c1418ae" />

---

## 📈 Feature Correlation & Selection

| Step | Purpose | Action |
| ---- | -------- | ------- |
| **Low-Variance Filter** | Remove features with <1% variability | Dropped non-informative columns |
| **High Correlation Filter** | Avoid multicollinearity (r > 0.9) | Retained key metrics like `resolution_days` |
| **Derived Features Review** | Flag redundant interactions | Retained meaningful engineered variables |

Result: Reduced noise, improved model interpretability, and enhanced generalization.

---

## ⚖️ Handling Class Imbalance

**Target Variable:** `satisfaction_label`  
- 0 → Unsatisfied / Low Satisfaction  
- 1 → Satisfied / High Satisfaction  

**Distribution:**
- Unsatisfied (0): 1,682  
- Satisfied (1): 1,087  

🧠 **Imbalance Solution:**  
Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance both classes during training.

---

## 🤖 Model Development

**Algorithms Used:**
1. Logistic Regression (Balanced Class Weights)  
2. XGBoost (Balanced Weights)  
3. Logistic Regression with SMOTE  
4. XGBoost with SMOTE  

**Train-Test Split:** 80% / 20% (Stratified)  
**Total Features:** 115  
**Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix  

---

## 🧪 Model Evaluation

| Model | Accuracy | F1 (Unsatisfied) | F1 (Satisfied) | Observation |
| ------ | -------- | ---------------- | --------------- | ------------ |
| Logistic Regression (Balanced) | 0.54 | 0.60 | 0.46 | Moderate recall for satisfied class |
| XGBoost (Balanced) | 0.52 | 0.60 | 0.40 | Slight bias toward majority class |
| Logistic Regression (SMOTE) | 0.53 | 0.60 | 0.43 | Improved recall for minority class |
| XGBoost (SMOTE) | 0.53 | 0.66 | 0.26 | Strong for majority class only |

📊 **Insight:**  
Logistic Regression with class balancing yielded **the most stable performance**, predicting both classes moderately well.

---

## 🔢 Regression Modeling — Predicting Resolution Time

**Target:** `resolution_days` (continuous variable)  

**Models Tested:**
- Linear Regression  
- Ridge Regression (L2)  
- Lasso Regression (L1)  
- Random Forest Regressor  
- XGBoost Regressor  
- LightGBM Regressor  
- Support Vector Regressor (SVR)  

**Best Model:** 🏆 **Lasso Regression**  
- Adjusted R² = **0.999995**  
- Excellent fit, minimal error  
- Linear models outperformed complex ensembles  
- Lasso effectively handled multicollinearity and feature selection

📉 **Interpretation:**  
Predicted vs. Actual scatter plot shows near-perfect alignment (y = x line), confirming high model accuracy.

<img width="1138" height="775" alt="image" src="https://github.com/user-attachments/assets/d99a1c8c-2584-4f12-bab6-f0d270ac83e1" />

<img width="1917" height="913" alt="image" src="https://github.com/user-attachments/assets/da36ab1f-0358-43ad-be72-1b890950b6e4" />
<img width="2862" height="1563" alt="image" src="https://github.com/user-attachments/assets/7484f67b-18fd-434a-b6fe-ffa98f1852db" />
<img width="2937" height="1612" alt="image" src="https://github.com/user-attachments/assets/da025fb8-d704-4ec4-a50c-ea6a51af675f" />


---

## 🧰 Tools & Technologies Used

| Tool / Library | Purpose |
| --------------- | -------- |
| 🐍 **Python** | Programming & modeling |
| 📓 **Jupyter Notebook** | Data analysis environment |
| 📊 **Pandas, NumPy** | Data manipulation |
| 📈 **Matplotlib, Seaborn** | Visualization |
| 🤖 **Scikit-learn, XGBoost, LightGBM** | Machine learning |
| 💬 **TextBlob, VADER** | Sentiment analysis |
| ⚖️ **SMOTE (Imbalanced-learn)** | Class balancing |
| 🔡 **TF-IDF (NLP)** | Text vectorization |
| 🧮 **SQL** | Data querying and preprocessing |

---

## 📁 Project Structure

```
customer-support-prediction/
├── app.py/
├── models/
│ ├── satisfaction_model.pkl # Logistic Regression or Voting Classifier for satisfaction prediction
│ ├── linear_regression_model.pkl # Linear/Lasso Regression for resolution time prediction
│ ├── desc_vectorizer.pkl # TF-IDF vectorizer for ticket description text
│ ├── res_vectorizer.pkl # TF-IDF vectorizer for resolution text
│ ├── feature_names.pkl # Saved list of final feature names
│ ├── model_features.pkl # Feature importance or encoded feature matrix
├── utils/
│   └── preprocessing.py        # Feature engineering & encoding
├── data/
│   └── sample_tickets.csv      # Example support data
├── requirements.txt
├── README.md
└── Customer-Satisfaction.ipynb      # Full pipeline development

```
---

## 🚀 How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/Customer-Satisfaction-Prediction.git
cd Customer-Satisfaction-Prediction
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Jupyter Notebook

```bash
jupyter notebook Customer-Satisfaction.ipynb
```

---



## 👨‍💻 Author

**M.Umesh Chandra**
📧 *[[metlaumeshchandra2005@gmail.com](mailto:metlaumeshchandra2005@gmail.com)]*
💼 *Data Scientist | Machine Learning Engineer | Climate Data Enthusiast*

---

⭐ **If you find this project insightful, please star the repository!**

```


