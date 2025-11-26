import os
import pandas as pd
import numpy as np

def process_file(num, filepath, desired_mean, desired_std):
    # Load the Excel file
    df = pd.read_excel(filepath, sheet_name=0, usecols='A:BU')
    # Standardize the numeric columns
    numeric_cols = df.select_dtypes(include='number')
    # Filter columns with at least two different values
    numeric_cols = numeric_cols.loc[:, numeric_cols.nunique() > 1]
    standardized_df = (numeric_cols - numeric_cols.mean(numeric_only=True)) / numeric_cols.std(numeric_only=True)
    # Scale and shift to achieve the desired mean and standard deviation
    adjusted_df = standardized_df * desired_std + desired_mean  
    # Combine with non-numeric columns
    df_updated = df.copy()
    df_updated[adjusted_df.columns] = adjusted_df
    df_updated.to_csv(filepath+".csv", sep=';', encoding='cp1251', index=False)
    # Save the original order of columns
    original_columns = df_updated.columns
    # Select only numeric columns
    df_numeric = df_updated.select_dtypes(include=[np.number])
    # Compute the result for each numeric column
    df_calc = df_numeric.mean(numeric_only=True)-df_numeric.median(numeric_only=True)
    # Replace non-numeric columns with -9999
    df_text = df_updated.select_dtypes(exclude=[np.number]).apply(lambda x: '-')
    df_concat = pd.concat([df_calc, df_text], axis=0)
    # Ensure the original order of columns is maintained
    result = pd.Series(df_concat, index=original_columns)    
    return result.to_frame(name=os.path.basename(filepath)).reset_index()

def median_file(num, filepath):
    # Load the Excel file
    df = pd.read_excel(filepath, sheet_name=0, usecols='A:BU')
    # Standardize the numeric columns
    numeric_cols = df.select_dtypes(include='number')
    # Filter columns with at least two different values
    numeric_cols = numeric_cols.loc[:, numeric_cols.nunique() > 1]
    # Save the original order of columns
    original_columns = df.columns
    # Select only numeric columns
    df_numeric = numeric_cols.select_dtypes(include=[np.number])
    # Compute the result for each numeric column
    df_calc = df_numeric.median(numeric_only=True)
    # Replace non-numeric columns with -9999
    df_text = numeric_cols.select_dtypes(exclude=[np.number]).apply(lambda x: '-')
    df_concat = pd.concat([df_calc, df_text], axis=0)
    # Ensure the original order of columns is maintained
    result = pd.Series(df_concat, index=original_columns)    
    return result.to_frame(name=os.path.basename(filepath)).reset_index()

def max_file(num, filepath):
    # Load the Excel file
    df = pd.read_excel(filepath, sheet_name=0, usecols='A:BU')
    # Standardize the numeric columns
    numeric_cols = df.select_dtypes(include='number')
    # Filter columns with at least two different values
    numeric_cols = numeric_cols.loc[:, numeric_cols.nunique() > 1]
    # Save the original order of columns
    original_columns = df.columns
    # Select only numeric columns
    df_numeric = numeric_cols.select_dtypes(include=[np.number])
    # Compute the result for each numeric column
    df_calc = df_numeric.median(numeric_only=True)+df_numeric.std(numeric_only=True)
    # Replace non-numeric columns with -9999
    df_text = numeric_cols.select_dtypes(exclude=[np.number]).apply(lambda x: '-')
    df_concat = pd.concat([df_calc, df_text], axis=0)
    # Ensure the original order of columns is maintained
    result = pd.Series(df_concat, index=original_columns)    
    return result.to_frame(name=os.path.basename(filepath)).reset_index()

# Desired mean and standard deviation
desired_mean = 100
desired_std = 30
tables = []
filenm = []
script_dir = os.path.dirname(__file__)
rootdir = script_dir+os.sep+'Data'
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith(".xlsx"):
            filepath = subdir + os.sep + file
            filenm.append(filepath)
            tables.append(process_file(len(filenm), filepath, desired_mean, desired_std))

# Merge the DataFrames on the index column
combined_df = pd.concat(tables, axis=1)
combined_df.to_csv(rootdir+os.sep+'div.csv', sep=';', encoding='cp1251', index=False)

print(f"\n\n\n")
tables2 = []
for filepath in filenm:
    tables2.append(median_file(len(tables2)+1, filepath))
# Merge the DataFrames on the index column
combined_df = pd.concat(tables2, axis=1)
combined_df.to_csv(rootdir+os.sep+'median.csv', sep=';', encoding='cp1251', index=False)

print(f"\n\n\n")
tables3 = []
for filepath in filenm:
    tables3.append(max_file(len(tables3)+1, filepath))
# Merge the DataFrames on the index column
combined_df = pd.concat(tables3, axis=1)
combined_df.to_csv(rootdir+os.sep+'max.csv', sep=';', encoding='cp1251', index=False)
