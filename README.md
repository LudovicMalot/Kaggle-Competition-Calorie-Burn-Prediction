# 🚀 Kaggle Competition: Calorie Burn Prediction 🏋️‍♂️

This project presents a solution for the 2025 Kaggle Playground Series competition. The goal was to predict workout calorie expenditure using physiological data. The final model is a **stacked ensemble of gradient boosting models** enhanced with advanced feature engineering and a final bias-correction layer.

---

## 🧠 What This Project Does

It follows a sophisticated, end-to-end machine learning pipeline to deliver highly accurate predictions:

1.  **Enriches Raw Data**: A custom transformation pipeline creates new, insightful features from base metrics, such as Body Mass Index (BMI), age-squared, and advanced heart rate statistics.
2.  **Optimizes Multiple Models**: It uses the **Optuna** framework to systematically find the best hyperparameters for three powerful gradient boosting models: LightGBM, XGBoost, and CatBoost.
3.  **Combines Predictions with Stacking**: A `StackingRegressor` acts as a "manager" model. It takes the predictions from the individual base models as input and learns how to combine them for a more accurate and robust final prediction.
4.  **Refines the Final Output**: A simple, final linear regression model is trained to correct any systematic bias in the ensemble's predictions, providing a last-mile performance boost.

## ⚙️ Technologies & Libraries

* **Core Stack**: Python, NumPy, Pandas
* **Machine Learning**: Scikit-learn (for Pipelines, Stacking, and Preprocessing)
* **Gradient Boosting**: LightGBM, XGBoost, CatBoost
* **Hyperparameter Tuning**: Optuna
* **Utilities**: Joblib (for saving model artifacts), Logging

---

## 🧪 Feature Engineering & Preprocessing

The foundation of the model's success lies in its comprehensive feature engineering, all encapsulated within a robust `scikit-learn` Pipeline to prevent data leakage.

### Architecture Highlights:

* **Custom Transformer**: A dedicated class `CustomFeatureEngineer` handles all new feature creation.
* **Key Engineered Features**:
    * **BMI Features**: `BMI`, `BMI_sq`, and `BMI_cat` (underweight, normal, etc.).
    * **Age-Based Features**: `Age_sq` and `Age_decade` to capture non-linear relationships.
    * **Heart Rate Analytics**: `HR_max_est`, `HR_reserve`, and `HR_pct_max` to contextualize heart rate.
    * **Interaction Terms**: `Weight * Duration` to model combined effects.
* **Preprocessing**: The pipeline automatically handles missing value imputation, one-hot encoding for categorical data, and `StandardScaler` for numeric features.

---

## 🤖 Stacked Ensemble Model

The core of the solution is a stacked ensemble that leverages multiple models.

### Architecture:

* **Base Models**:
    * `Ridge`: A simple linear baseline.
    * `LightGBM`: Optimized with Optuna.
    * `XGBoost`: Optimized with Optuna.
    * `CatBoost`: Optimized with Optuna.
* **Hyperparameter Tuning**: Each boosting model was tuned using **Optuna** over 20-50 trials with 5-fold cross-validation to find the most effective parameters.
* **Meta-Model**: A `Ridge` regressor serves as the final estimator, learning the optimal weights to assign to each base model's prediction.

### 🎯 Bias Correction Layer

A final `LinearRegression` model was trained on the out-of-fold predictions from the validation set. This unique step acts as a fine-tuning mechanism, correcting for any small, systematic errors and ensuring the predictions are perfectly calibrated.

## 🏆 Results & Performance

The model achieved a top-tier rank, demonstrating its high accuracy and effectiveness.

| Metric                    | Value                                    |
| ------------------------- | ---------------------------------------- |
| 🏁 **Final Rank**          | **370/4316** (Top 9%)                   |
| 🎯 **Final Score (RMSLE)** | `0.05866`  (top is 0.05841)             |

---

## 🧠 Skills Demonstrated

* Advanced Feature Engineering for Tabular Data
* End-to-End Machine Learning Pipelines (`scikit-learn`)
* Hyperparameter Optimization (Optuna)
* Advanced Ensemble Methods (Stacking)
* Model Evaluation, Calibration, and Refinement
* Best Practices in Reproducible ML (Pipelines, Artifact Saving)

---

## 📚 Dataset Source

* **[Kaggle Playground Series](https://www.kaggle.com/competitions/playground-series-s5e5/overview)**
    * This competition is part of Kaggle's monthly series designed for practicing and honing machine learning skills on approachable, real-world datasets.

---

## 🙋‍♂️ Author

👋 Hi! I'm **Ludovic Malot**, a French engineer with a passion for building effective and creative machine learning solutions. This project was a fantastic opportunity to apply and refine advanced ensembling techniques in a highly competitive environment.

Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/ludovic-malot/) or drop a ⭐ if you found this project insightful!
