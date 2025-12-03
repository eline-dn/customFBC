import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# define some helper functions

# concat several datasets into one, adding a condtion col and ensuring it is clean
def concat_conditions(data_path, conditions, csv_pattern="opti_design_stats"):
  """
  pata_path = folder where each conditions' folder is located
  list of the conditions's name
  pattern to catch the csv fiel containing the info in our dcondition file
  """
  for csv in glob.glob(f"{data_path}/*/{csv_pattern}.csv") :
    # check condition name 
    if "/".split(csv)[-2] in conditions:
      condition="/".split(csv)[-2]
      print("processing condition", condition)
    else:
      continue
    df = pd.read_csv(csv)
    df['condition'] = condition
    if df.iloc[:, 0].astype(str).str[:len(condition)] != condition:
      term=df.iloc[:, 0].astype(str).str[:len(condition)]
      print(f"Warning: mismatching condition ({condition}) and binder terminology ({term}), please check data")
    # remove any non data lines: eg additionnal headers
    if # nothing in fiirst col:
    
