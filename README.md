nsurance Cost Prediction 🏥📊
Overview
Insurance companies need to accurately predict the cost of health insurance for individuals to set premiums appropriately. Traditional methods relying on broad actuarial tables often fail to account for nuanced individual differences. This project leverages Machine Learning techniques to predict insurance costs tailored to individual profiles, aiming to enhance pricing precision, increase competitiveness, and improve customer satisfaction.
Dataset Description
The dataset comprises 986 observations with no missing or duplicated values. It includes the following 11 attributes (all integer types):
Age: 18 to 66 years.
Height: 145 cm to 188 cm.
Weight: 51 kg to 132 kg.
Diabetes: Binary (0 or 1).
BloodPressureProblems: Binary (0 or 1).
AnyTransplants: Binary (0 or 1).
AnyChronicDiseases: Binary (0 or 1).
KnownAllergies: Binary (0 or 1).
HistoryOfCancerInFamily: Binary (0 or 1).
NumberOfMajorSurgeries: 0 to 3 surgeries.
PremiumPrice: The target variable, ranging from 15,000 to 40,000.
Project Workflow & Key Findings
1. Data Quality & Exploratory Data Analysis (EDA)
A comprehensive custom Object-Oriented EDA framework was built to analyze the data distributions and relationships.
Demographic Insights: Within the dataset, 42% have diabetes, 47% experience blood pressure problems, 18% have chronic diseases, and 52% have had at least one major surgery. Organ transplants (6%) and a family history of cancer (12%) are relatively rare.
Distributions: Weight exhibits a slight right skew, indicating the presence of outliers (13 identified via Z-Score).
Correlations: Age and Premium Price are strongly positively correlated ($r = 0.70$).
2. Statistical Hypothesis Testing
Rigorous statistical tests were conducted to determine the significance of features against the PremiumPrice.
Normality (Shapiro-Wilk): Tests revealed that the numerical features (Age, Height, Weight, PremiumPrice) do not follow a normal distribution.
Feature Significance (Kruskal-Wallis): Since the data is non-parametric, Kruskal-Wallis tests were used. Diabetes, Blood Pressure, Transplants, Chronic Diseases, Cancer History, and Major Surgeries all have a statistically significant impact on the Premium Price. Known Allergies showed no significant impact and acts as an independent variable.
Feature Associations (Chi-Square): * Diabetes is strongly associated with Blood Pressure problems, Chronic Diseases, and Major Surgeries.
Transplants and Family Cancer History act largely as isolated factors without significant ties to other health markers.
3. Preprocessing & Feature Engineering
Feature Engineering: A BMI feature was initially calculated using Weight and Height.
Multicollinearity Handling: Variance Inflation Factor (VIF) analysis was conducted. BMI was identified with an extremely high VIF (85.58) and was subsequently dropped to prevent multicollinearity, leaving all remaining features well below the VIF threshold of 5.
Scaling: Data was split 80/20 for training and testing, and scaled using MinMaxScaler.
4. Machine Learning Modeling & Evaluation
Various models were tested to evaluate predictive performance on the target PremiumPrice.
Linear Baselines
Model
Train R2
Test R2
Test Adjusted R2
Linear Regression
0.6221
0.7134
0.6981
Lasso Regression ( $\alpha=0.01$ )
0.6219
0.7133
0.7097
Ridge Regression ( $\alpha=1$ )
0.6216
0.7110
0.7073

Diagnostic Note: Residuals for the linear regression models are not normally distributed, and Goldfeld-Quandt tests indicated the presence of heteroskedasticity (increasing variance in errors). A 4-fold cross-validation yielded an average $R^2$ of ~0.6029.
Non-Linear Baselines
KNN Regressor: Iterative testing found the optimal neighbors at $k=4$. However, the model underperformed compared to linear baselines, yielding a Test $R^2$ of 0.5879.
