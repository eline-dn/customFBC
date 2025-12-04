import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.stats import pearsonr
# define some helper functions

# concat several datasets into one, adding a condtion col and ensuring it is clean
def concat_conditions(data_path, conditions, csv_pattern="opti_design_stats"):
  """
  pata_path = folder where each conditions' folder is located
  list of the conditions's name
  pattern to catch the csv fiel containing the info in our dcondition file
  """
  df_list=list()
  for csv in glob.glob(f"{data_path}/*/{csv_pattern}*.csv") :
    # check condition name 
    print(csv)
    if csv.split("/")[-2] in conditions:
      condition=csv.split("/")[-2]
      print("processing condition", condition)
    else:
      continue
    df = pd.read_csv(csv)
    df['condition'] = condition
    print(df.head())
    if df.iloc[1, 0][:len(condition)] != condition:
      term=df.iloc[1, 0][:len(condition)]
      print(f"Warning: mismatching condition ({condition}) and binder terminology ({term}), please check data")
    # remove any non data lines: eg additionnal headers
    #df[0].replace('', np.nan, inplace=True)
    df.iloc[:,0]=df.iloc[:, 0].replace('', np.nan)
    first_column_name = df.columns[0]
    df=df.dropna(subset=[first_column_name])# nothing in fiirst col
    # Define columns to exclude from numeric conversion
    exclude_columns = [first_column_name, 'models', 'condition'] # 'condition' is also typically a string
    # Get all column names from the DataFrame
    all_columns = df.columns.tolist()
    # Filter out the excluded columns to get the list of columns to convert
    columns_to_convert = [col for col in all_columns if col not in exclude_columns]
    for col in columns_to_convert:
      # Convert column to numeric, coercing errors to NaN
      df[col] = pd.to_numeric(df[col], errors='coerce')
    df_list.append(df)
  return(pd.concat(df_list, ignore_index=True))


def correlation(df, col_x, col_y, condition_col):
  plt.figure(figsize=(5, 5))

  ax = plt.gca() # Get current axes
  sns.scatterplot(data=df, x=col_x, y=col_y, hue=condition_col, ax=ax)
  plt.xlabel(col_x)
  plt.ylabel(col_y)
  plt.title(f'Scatter plot of {col_x} vs {col_y} (colored by {condition_col})')

  # Calculate and display Pearson correlation
  r, _ = pearsonr(df[col_x], df[col_y])
  ax.text(0.05, 0.95, f'Pearson R: {r:.2f}', transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

  plt.tight_layout()
  #plt.show()
  plt.savefig(f"Scatter_{col_x}_vs_{col_y}.png")
  plt.close()


def plot_distribution(df, col, conditions, condition_col):
  """col is a str with the col's name to plot
  condition_col is a str with the name of of the condition column
  conditions is a list of str with the name of the cols to plot"""
  # Plot distributions
  plt.figure(figsize=(10, 6))
  
  for condition in conditions:
    sns.histplot(data=df[(df[condition_col] == condition)], x=col, kde=True, label=condition, stat='density') # Changed stat to density
  
  plt.xlabel(f'Value of {col}')
  plt.ylabel('Density') # Changed label to Density
  plt.title(f'Distribution of {col} ')
  plt.legend()
  #plt.show()
  plt.savefig(f"distribution_{col}_acr_{condition_col}.png")
  plt.close()





#---------------------------------------
conditions=list(("ct0","ct5bis", "ct7","ct8","ct9","ct10"))
df=concat_conditions(data_path="/work/lpdi/users/eline/binderdesign", conditions=conditions, csv_pattern="opti_design_stats")
df.head()
correlation(
    df=df,
    col_x='iptm_contrib_A_barrel',
    col_y='1_specific_empty_i_pTM',
    condition_col='condition'
)

# You can call it again for 'iptm_contrib_A_helix' vs '1_specific_plugged_i_pTM' if you want two separate plots
correlation(
    df=df,
    col_x='iptm_contrib_A_helix',
    col_y='1_specific_plugged_i_pTM',
    condition_col='condition'
)

plot_distribution(df=df, col='1_specific_empty_i_pTM', conditions=conditions, condition_col='condition')
plot_distribution(df=df, col='1_specific_plugged_i_pTM', conditions=conditions, condition_col='condition')
