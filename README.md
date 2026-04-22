# Insurance Cost Prediction 🏥📊

## Overview

Insurance companies need to accurately predict the cost of health insurance for individuals to set premiums appropriately. Traditional methods relying on broad actuarial tables often fail to account for nuanced individual differences. This project leverages Machine Learning techniques to predict insurance costs tailored to individual profiles, aiming to enhance pricing precision, increase competitiveness, and improve customer satisfaction.

## Dataset Description

The dataset comprises 986 observations with no missing or duplicated values. It includes the following 11 attributes (all integer types):

- **Age:** 18 to 66 years.
- **Height:** 145 cm to 188 cm.
- **Weight:** 51 kg to 132 kg.
- **Diabetes:** Binary (0 or 1).
- **BloodPressureProblems:** Binary (0 or 1).
- **AnyTransplants:** Binary (0 or 1).
- **AnyChronicDiseases:** Binary (0 or 1).
- **KnownAllergies:** Binary (0 or 1).
- **HistoryOfCancerInFamily:** Binary (0 or 1).
- **NumberOfMajorSurgeries:** 0 to 3 surgeries.
- **PremiumPrice:** The target variable, ranging from 15,000 to 40,000.

## Project Workflow & Key Findings

### 1\. Data Quality & Exploratory Data Analysis (EDA)

A comprehensive custom Object-Oriented EDA framework was built to analyze the data distributions and relationships.

- **Demographic Insights:** Within the dataset, 42% have diabetes, 47% experience blood pressure problems, 18% have chronic diseases, and 52% have had at least one major surgery. Organ transplants (6%) and a family history of cancer (12%) are relatively rare.
- **Distributions:** Weight exhibits a slight right skew, indicating the presence of outliers (13 identified via Z-Score).
- **Correlations:** Age and Premium Price are strongly positively correlated (\$r = 0.70\$).

### 2\. Statistical Hypothesis Testing

Rigorous statistical tests were conducted to determine the significance of features against the PremiumPrice.

- **Normality (Shapiro-Wilk):** Tests revealed that the numerical features (Age, Height, Weight, PremiumPrice) do _not_ follow a normal distribution.
- **Feature Significance (Kruskal-Wallis):** Since the data is non-parametric, Kruskal-Wallis tests were used. Diabetes, Blood Pressure, Transplants, Chronic Diseases, Cancer History, and Major Surgeries all have a statistically significant impact on the Premium Price. **Known Allergies showed no significant impact** and acts as an independent variable.
- **Feature Associations (Chi-Square):** \* _Diabetes_ is strongly associated with Blood Pressure problems, Chronic Diseases, and Major Surgeries.
  - _Transplants_ and _Family Cancer History_ act largely as isolated factors without significant ties to other health markers.

### 3\. Preprocessing & Feature Engineering

- **Feature Engineering:** A BMI feature was initially calculated using Weight and Height.
- **Multicollinearity Handling:** Variance Inflation Factor (VIF) analysis was conducted. BMI was identified with an extremely high VIF (85.58) and was subsequently dropped to prevent multicollinearity, leaving all remaining features well below the VIF threshold of 5.
- **Scaling:** Data was split 80/20 for training and testing, and scaled using MinMaxScaler.

### 4\. Machine Learning Modeling & Evaluation

Various models were tested to evaluate predictive performance on the target PremiumPrice.

#### Linear Baselines

| **Model**                                 | **Train R2** | **Test R2** | **Test Adjusted R2** |
| ----------------------------------------- | ------------ | ----------- | -------------------- |
| **Linear Regression**                     | 0.6221       | 0.7134      | 0.6981               |
| ---                                       | ---          | ---         | ---                  |
| **Lasso Regression ( \$\\alpha=0.01\$ )** | 0.6219       | 0.7133      | 0.7097               |
| ---                                       | ---          | ---         | ---                  |
| **Ridge Regression ( \$\\alpha=1\$ )**    | 0.6216       | 0.7110      | 0.7073               |
| ---                                       | ---          | ---         | ---                  |

_Diagnostic Note:_ Residuals for the linear regression models are not normally distributed, and Goldfeld-Quandt tests indicated the presence of heteroskedasticity (increasing variance in errors). A 4-fold cross-validation yielded an average \$R^2\$ of ~0.6029.

#### Non-Linear Baselines

- **KNN Regressor:** Iterative testing found the optimal neighbors at \$k=4\$. However, the model underperformed compared to linear baselines, yielding a Test \$R^2\$ of **0.5879**.

## Tree-Based Models & Ensemble Techniques

As we moved beyond linear baselines and distance-based algorithms, we explored tree-based models and their ensemble counterparts. These models are particularly well-suited for tabular data and capturing non-linear relationships without the need for extensive feature scaling.

### 1\. Decision Trees

We began with a foundational Decision Tree Regressor. To prevent the model from memorizing the training data, we performed a 10-fold cross-validation search across maximum tree depths ranging from 2 to 14.

- **Performance:** Deeper trees (depths 10+) severely overfit the training data, achieving nearly 100% training accuracy while test accuracy stagnated. We identified an optimal max_depth of **4**.
- **Scores:** Using this optimal depth on our holdout set, the model achieved a Training \$R^2\$ of **76.28%** and an impressive Testing \$R^2\$ of **83.14%**.
- **Feature Importance:** The model's predictions were overwhelmingly driven by **Age** (contributing 75.9% to the predictive power), followed by **AnyTransplants** (12.3%) and **Weight** (3.8%).

### 2\. Random Forest

To build upon the Decision Tree and mitigate the risk of overfitting, we implemented a Random Forest Regressor. We utilized GridSearchCV (with 2-fold cross-validation over 144 candidate combinations) to rigorously tune hyperparameters.

- **Optimal Parameters:** The grid search identified the best configuration as max_depth=20, min_samples_leaf=2, min_samples_split=5, and n_estimators=400.
- **Scores:** This robust ensemble achieved a highly stable best cross-validation \$R^2\$ score of **~74.0%**, demonstrating strong generalization capabilities across different data splits.

### 3\. Advanced Boosting Algorithms

In addition to bagging techniques like Random Forest, we also structured pipelines for advanced Gradient Boosting algorithms, including **Standard Gradient Boosting**, **XGBoost**, and **LightGBM**. These models utilized similar exhaustive grid search approaches (tuning parameters like num_leaves, learning_rate, and n_estimators) to sequentially minimize prediction errors and push the boundaries of model accuracy.

## Conclusion: Optimal Model Selection

Predicting insurance premiums requires a delicate balance between high accuracy, interpretability, and consistent generalization to unseen patient profiles.

After evaluating a diverse suite of machine learning algorithms-ranging from Linear and Ridge/Lasso Regression to K-Nearest Neighbors and Advanced Gradient Boosting-**Random Forest emerged as the optimal model for this insurance cost prediction task.** While a solitary, shallow Decision Tree achieved high individual hold-out scores, it carried an inherent structural risk of variance and overfitting (as observed in depths beyond 4). By leveraging the Random Forest ensemble, we effectively reduced this variance. The Random Forest model achieved a highly stable, cross-validated \$R^2\$ score of roughly **74%**, successfully navigating the complex, non-linear interactions between critical patient features like Age, Transplant History, and Weight.

Ultimately, the Random Forest model provides the insurance provider with a highly reliable, robust, and scalable predictive engine. It enhances pricing precision, ensuring that premiums remain highly competitive while accurately reflecting the individualized health risks of the policyholders.
