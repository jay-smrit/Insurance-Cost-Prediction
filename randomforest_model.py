import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

insurance_df = pd.read_csv("insurance.csv")

X = insurance_df.drop('PremiumPrice', axis=1)
y = insurance_df['PremiumPrice']

rf_model = RandomForestRegressor(
    max_depth= 20,
    min_samples_leaf= 2,
    min_samples_split= 5,
    n_estimators= 400,
    random_state=42
)

rf_model.fit(X, y)

with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)