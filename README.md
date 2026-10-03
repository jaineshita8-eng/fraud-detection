# 💳 Credit Card Fraud Detection

An end-to-end machine learning project that detects fraudulent credit card transactions on a highly imbalanced dataset (only **0.17%** of transactions are fraud), compares multiple models, and serves the best one through an interactive **Streamlit** app.

**🔗 Live demo:** [your-app-name.streamlit.app](https://your-app-name.streamlit.app)

![App screenshot](assets/app_screenshot.png)

---

## 📌 Problem

Banks lose billions every year to card fraud, but fraud is rare: in this dataset only 492 of 284,807 transactions are fraudulent. A model that labels *everything* as legitimate would score **99.8% accuracy while catching zero fraud**. This project shows how to build a model that actually catches fraud, and why accuracy is the wrong yardstick for this kind of problem.

## 📊 Dataset

- **Source:** [Credit Card Fraud Detection (ULB) on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions made by European cardholders over two days in 2013
- **Features:** `V1–V28` (PCA-transformed, anonymized), `Time`, `Amount`
- **Target:** `Class` (0 = legitimate, 1 = fraud)
- **Class balance:** 99.83% legitimate / 0.17% fraud

> The raw `creditcard.csv` (~150 MB) is not included in this repo. Download it from the Kaggle link above and place it in a `data/` folder to rerun the notebook.

## 🛠️ Approach

1. **EDA:** confirmed the extreme class imbalance, inspected the `Amount` distribution, and verified there are no missing values.
2. **Preprocessing:** scaled `Amount` with `StandardScaler`, engineered an `Hour` feature from `Time`, and dropped the raw `Time` column.
3. **Stratified split:** 80/20 train/test with `stratify=y` so both sets keep the 0.17% fraud ratio. The test set was never resampled or tuned on.
4. **Baselines on raw data:** Logistic Regression and Random Forest, to show how misleading accuracy is.
5. **Imbalance handling:**
   - **SMOTE** applied to the **training set only** (applying it before the split would leak synthetic copies into the test set).
   - **Class weighting** via XGBoost's `scale_pos_weight`.
6. **Evaluation:** Precision, Recall, F1, ROC-AUC, and PR-AUC. Accuracy is deliberately not used for model selection.
7. **Deployment:** the best model is saved with `joblib` and served through Streamlit with an adjustable decision threshold.

## 📈 Results

Evaluated on the held-out, stratified test set (56,962 transactions, ~98 frauds).

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression (raw) | 0.xx | 0.xx | 0.xx | 0.xx | 0.xx |
| Random Forest (raw) | 0.xx | 0.xx | 0.xx | 0.xx | 0.xx |
| Random Forest + SMOTE | 0.xx | 0.xx | 0.xx | 0.xx | 0.xx |
| XGBoost + SMOTE | 0.xx | 0.xx | 0.xx | 0.xx | 0.xx |
| XGBoost + class weights | 0.xx | 0.xx | 0.xx | 0.xx | 0.xx |

> 🔧 *Replace the `0.xx` values with the numbers from your own results table.*

### Key takeaways

- **Accuracy was ~99.9% for every model**, including ones that missed a large share of fraud, so it is useless for comparing them.
- **Recall vs. precision is a business trade-off:** a missed fraud costs money, while a false alarm annoys a customer. The app's threshold slider lets you explore this trade-off directly.
- **PR-AUC is more informative than ROC-AUC** here, because ROC-AUC can look excellent even when precision is mediocre on rare-event data.
- *(Add your own finding here, e.g. which model won and by how much.)*

## 🚀 Run it locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/fraud-detection.git
cd fraud-detection

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

To reproduce the analysis, download `creditcard.csv` into `data/` and open `notebooks/fraud_detection.ipynb`.

## 🖥️ Using the app

- **Try a sample transaction:** pick a pre-loaded transaction (a mix of real fraud and legitimate cases) and get a Fraud / Legit verdict with a probability.
- **Upload CSV:** score many transactions at once (the file must have the same columns as `sample_transactions.csv`).
- **Threshold slider:** raise it to reduce false alarms, lower it to catch more fraud.

> The `V1–V28` features are anonymized PCA components, so the app works from sample rows and CSV uploads rather than hand-typed inputs.

## 📁 Project structure

```
fraud-detection/
├── app.py                      # Streamlit app
├── model.joblib                # Trained model
├── sample_transactions.csv     # Demo transactions for the app
├── notebooks/
│   └── fraud_detection.ipynb   # EDA, modelling, evaluation
├── assets/
│   └── app_screenshot.png
├── requirements.txt
└── README.md
```

## 🧰 Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · imbalanced-learn · Matplotlib · Seaborn · Streamlit · joblib

## ⚠️ Limitations and next steps

- The data is from 2013 and covers only two days, so real-world fraud patterns may differ.
- Features are anonymized, which limits interpretability. With raw features, SHAP could explain individual predictions.
- The decision threshold was tuned on the test set for demonstration; a production setup would use a separate validation set.
- Next steps: cross-validated hyperparameter tuning, cost-sensitive evaluation using actual fraud-loss amounts, and SHAP explanations.

## 👤 Author

**Your Name**
[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)

## 📄 License

This project is licensed under the MIT License. The dataset is released under the [Open Database License](https://opendatacommons.org/licenses/odbl/1-0/) by its original authors (Worldline and the ULB Machine Learning Group).
