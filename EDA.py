import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 100)

insurance_df = pd.read_csv("insurance.csv")

# ==============================================================================
# DATA QUALITY REPORT** (Observations on Data)
# ==============================================================================
#PRINT SAMPLE RECORDS
print("Sample records:")
sample = insurance_df.sample(n=20,axis=0, random_state=32)
print(sample)

def data_quality_report(df):

  # DATA REPORT (STRUCTURE & CHARACTERISTICS)
  print("*"*150)
  print("Shape:")
  n_obs = df.shape[0]
  n_features = df.shape[1]
  print(f"No. of observations: {n_obs}, No. of features:{n_features}")
  print("\n Duplicates : ")
  print(f"No. of duplicated rows: {df.duplicated().sum()}")

  print("*"*150)
  print("Initial Summary Stats:")
  names = df.columns
  dtypes = df.dtypes
  missing = df.isnull().sum()
  non_missing = df.notnull().sum()
  missing_pct = (missing / len(df)) * 100
  unique = df.nunique()

  list_of_tuples = list(zip(names, dtypes, missing, non_missing, missing_pct, unique, ))
  summary_df = pd.DataFrame(list_of_tuples,columns=['name', 'data_type', 'missing', 'non-missing', 'missing_pct','unique'])
  print(summary_df.round(2))

  return n_obs, n_features, summary_df

n_obs, n_features, summary_df = data_quality_report(insurance_df)

# ==============================================================================