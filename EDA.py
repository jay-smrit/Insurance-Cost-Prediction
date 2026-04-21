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
# 1. DATA QUALITY REPORT (Observations on Data)
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

num_vars = list(['Age',
                 'Height',
                 'Weight',
                 'PremiumPrice'])

cat_vars = list(['Diabetes',
                 'BloodPressureProblems',
                 'AnyTransplants',
                 'AnyChronicDiseases',
                 'KnownAllergies',
                 'HistoryOfCancerInFamily',
                 'NumberOfMajorSurgeries',])

target = 'PremiumPrice'

# ==============================================================================
# 2. UNIVARIATE ANALYSIS
# ==============================================================================
class EDAUnivariateAnalysis:
  """
  Comprehensive EDA Framework for Univariate Analysis
  """
  def __init__(self, df, cat_vars, num_vars):
    self.df = df.copy()
    self.cat_vars = cat_vars
    self.num_vars = num_vars

# NUMERICAL STATS
  def numeric_analysis(self):
    print("Numerical Analysis:")
    print("*"*250)

    # num_vars = self.num_vars
    num_df = self.df[self.num_vars]

    numeric_stats = num_df.describe(percentiles=[.01, 0.05, 0.1, .25, .5, .75, .8, .9, .95, .99]).T.round(2)
    numeric_stats.rename(columns={'index' : 'name'},inplace=True)

    numeric_stats['skew'] = num_df.skew()
    numeric_stats['kurt'] = num_df.kurt()

    return numeric_stats.round(2)

    #OUTLIER DETECTION
  def outlier_detection(self):
    print("*"*250)
    print("Outlier Detection:")
    num_df = self.df[self.num_vars]
    outlier_lst = []
    for col in self.num_vars:
      mean = num_df[col].mean()
      std = num_df[col].std()
      Q1 = num_df[col].quantile(0.25)
      Q3 = num_df[col].quantile(0.75)
      IQR = Q3 - Q1
      IQR_LB = Q1 - 1.5 * IQR
      IQR_UB = Q3 + 1.5 * IQR
      ZScore_LB = mean - 3 * std
      ZScore_UB = mean + 3 * std
      IQR_outlier_cnt = ((num_df[col] < IQR_LB) | (num_df[col] > IQR_UB)).sum()
      ZScore_outlier_cnt = ((num_df[col] < ZScore_LB) | (num_df[col] > ZScore_UB)).sum()

      outlier_dict = {
          'IQR' : IQR,
          'IQR_LB' : IQR_LB,
          'IQR_UB' : IQR_UB,
          'ZScore_LB' : ZScore_LB,
          'ZScore_UB' : ZScore_UB,
          'IQR_outlier_cnt' : IQR_outlier_cnt,
          'ZScore_outlier_cnt' : ZScore_outlier_cnt,
      }
      outlier_lst.append(outlier_dict)

    outlier_stats = pd.DataFrame(outlier_lst,index=self.num_vars)

    outliers_IQR = (
        (num_df[self.num_vars] < outlier_stats['IQR_LB']) | (num_df[self.num_vars] > outlier_stats['IQR_UB'])).any(axis=1).sum()

    outliers_ZScore = (
        (num_df < outlier_stats['ZScore_LB']) | (num_df > outlier_stats['ZScore_UB'])).any(axis=1).sum()

    print("\nTotal outliers using IQR : ", outliers_IQR)
    print("Total outliers using ZScore : ", outliers_ZScore)

    return outlier_stats.round(2)

  def graphical_num(self):
    print("="*250)
    print("Histogram / Box plot for Continous Variables: ")
    num_vars = self.num_vars
    for col in num_vars:
      fig, axes = plt.subplots(1, 2, figsize=(12,4))

      #plot histogram
      sns.histplot(data=self.df, x=col, kde=True, stat='density',line_kws={'linewidth': 3}, ax=axes[0])
      axes[0].set_title(f"Histogram of {col}")
      axes[0].set_xlabel("")

      #plot boxplot
      sns.boxplot(data=self.df, x = col, ax=axes[1])
      axes[1].set_title(f"Box plot of {col}")
      axes[1].set_xlabel("")

      plt.tight_layout()

    plt.show()

###################################################################################################

  def categoric_analysis(self):
    print("*"*150)
    print("Categorical Analysis:")
    category_df = pd.DataFrame()
    for col in self.cat_vars:
      temp_df_1 = self.df[col].value_counts(dropna=False).reset_index()
      temp_df_2 = self.df[col].value_counts(dropna=False,normalize=True).reset_index()

      print("*"*150)
      # print(f"Frequency Distribution for {col}:")
      temp_df = pd.merge(temp_df_1, temp_df_2, how='inner', on=col)
      print(temp_df.round(2))

    print("="*250)
    print("Count Plot for Discrete variables: ")
    cat_vars = self.cat_vars
    cols = 2
    rows = (len(cat_vars) // cols) + (len(cat_vars) % cols)

    fig, axs = plt.subplots(nrows= rows, ncols= cols, figsize=(20,16))
    fig.subplots_adjust(top=1.3)
    count = 0
    for row in range(rows):
      for col in range(cols):
        sns.countplot(data=self.df, x=cat_vars[count], ax=axs[row, col], palette='Set2',legend=False)
        axs[row, col].set_title(f"{cat_vars[count]}", pad=12, fontsize=16)
        axs[row, col].set_xlabel(None)
        axs[row, col].set_ylabel(None)
        axs[row, col].tick_params(axis='x', labelrotation=45)
        count += 1
        if count == len(cat_vars):
          break

    plt.show()

  ###################################################################################################
eda_univariate = EDAUnivariateAnalysis(insurance_df, cat_vars, num_vars)

numeric_stats = eda_univariate.numeric_analysis()
print(numeric_stats)

outlier_stats = eda_univariate.outlier_detection()
print(outlier_stats)

eda_univariate.graphical_num()

eda_univariate.categoric_analysis()
# ==============================================================================