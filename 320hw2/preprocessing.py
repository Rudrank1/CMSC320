import pandas as pd
import numpy as np
from difflib import SequenceMatcher

def standardize_column_names(df):
    # Standardize column names across datasets by converting gender-specific terms to neutral terms
    replacements = {
        ' says ': ' say ',
        ' needs ': ' need ',
        ' boyfriend ': ' partner ',
        ' boyfriend, ': ' partner, ',
        ' girlfriend ': ' partner ',
        ' girlfriend, ': ' partner, ',
        ' son ': ' child ',
        ' son. ': ' child. ',
        ' daughter ': ' child ',
        ' daughter. ': ' child. ',
        ' brother ': ' sibling ',
        ' brother\'s ': ' sibling\'s ',
        ' brother. ': ' sibling. ',
        ' sister ': ' sibling ',
        ' sister\'s ': ' sibling\'s ',
        ' sister. ': ' sibling. ',
        ' mom ': ' parent ',
        ' father ': ' parent ',
        ' father-in-law ': ' parent-in-law ',
        ' mother ': ' parent ',
        ' mother-in-law ': ' parent-in-law ',
        ' husband ': ' spouse ',
        ' wife ': ' spouse ',
        ' he ': ' they ',
        ' she ': ' they ',
        ' him ': ' them ',
        ' her ': ' them ',
        ' his ': ' their ',
        ' hers ': ' their ',
        ' hes ': ' theyre ',
        ' shes ': ' theyre ',
        ' himself ': ' themself ',
        ' herself ': ' themself ',
        ' man ': ' person ',
        ' woman ': ' person ',
        ' men ': ' people ',
        ' women ': ' people ',
        ' he\'s ': ' they\'ve ',
        ' she\'s ': ' they\'ve ',
        ' he\'d ': ' they\'d ',
        ' she\'d ': ' they\'d ',
        ' he\'ll ': ' they\'ll ',
        ' she\'ll ': ' they\'ll ',
        ' herself, ': ' themself, ',
        ' himself, ': ' themself, ',
        ' they is ': ' they are ',
        ' wasnt ': ' weren\'t ',
        ' treats ': ' treat ',
        ' they was ': ' they were ',
        ' they has ': ' they have ',
        ' they doesnt ': ' they don\'t ',
        ' they didnt ': ' they didn\'t ',
        'he calls': 'they call',
        ' asks ': " ask "
    }
    
    new_columns = {}
    for col in df.columns:
        new_col = col.lower()
        for old, new in replacements.items():
            if old in new_col:
                new_col = new_col.replace(old, new)
        new_columns[col] = new_col
    
    return df.rename(columns=new_columns)

def similar(a, b, threshold=0.8):
    # Return True if strings a and b are similar enough
    return SequenceMatcher(None, a, b).ratio() > threshold

def merge_similar_columns(df):
    # Merge columns that are asking the same question with slightly different wording
    columns = list(df.columns)
    
    similar_cols = {}
    
    for i, col1 in enumerate(columns):
        if col1 not in similar_cols:
            similar_cols[col1] = [col1]
            for col2 in columns[i+1:]:
                if similar(col1, col2):
                    similar_cols[col1].append(col2)
    
    for main_col, similar_group in similar_cols.items():
        if len(similar_group) > 1:
            for col in similar_group[1:]:
                df[main_col] = df[main_col].combine_first(df[col])
                if col in df.columns:
                    df = df.drop(columns=[col])
    
    return df

def clean_dataset(df, genders_reversed=False):
    # Clean a single dataset
    df = standardize_column_names(df)
    
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(r'\s+', '_')
    df.columns = df.columns.str.replace(r'[^\w\s-]', '')
    
    df = df.replace(['', ' ', 'NA', 'N/A', 'n/a', 'na', 'null', 'NULL', 'None', 'none'], np.nan)
    
    return df

def drop_empty_columns(df, threshold=0.25):
    # Drop columns that have too many missing values (default threshold is 25% missing)
    missing_percentages = df.isnull().mean()
    columns_to_drop = missing_percentages[missing_percentages > threshold].index
    df = df.drop(columns=columns_to_drop)
    
    return df

def drop_rows_with_missing_values(df):
    # Drop any rows that have missing values in any column
    df = df.dropna()
    
    return df

def encode_jerk_columns(df):
    # Perform manual label encoding on all columns containing 'jerk' in their name
    jerk_columns = [col for col in df.columns if 'jerk' in col.lower()]

    jerk_mapping = {
        'Not a jerk': 0,
        'Mildly a jerk': 1,
        'Strongly a jerk': 2
    }
    
    for col in jerk_columns:
        df[col] = df[col].map(jerk_mapping)
    
    return df

# Read the datasets
df_2024 = pd.read_csv('datasets/Dataset Generation (2024) (Responses) - Form Responses 1.csv')
df_fardina = pd.read_csv('datasets/Dataset Generation (Fardina) (Responses) - Form Responses 1.csv')
df_max = pd.read_csv('datasets/Dataset Generation (Max) (Responses) - Form Responses 1.csv')
df_spring2025 = pd.read_csv('datasets/Dataset Generation (Spring 2025) (Responses) - Form Responses 1.csv')

# Clean each dataset
df_2024 = clean_dataset(df_2024, genders_reversed=True)
df_fardina = clean_dataset(df_fardina)
df_max = clean_dataset(df_max)
df_spring2025 = clean_dataset(df_spring2025, genders_reversed=True)

# Preprocess and save the combined dataset
combined_df = pd.concat([df_2024, df_fardina, df_max, df_spring2025], ignore_index=True)
combined_df = merge_similar_columns(combined_df)
combined_df = drop_empty_columns(combined_df)
combined_df = drop_rows_with_missing_values(combined_df)
combined_df = encode_jerk_columns(combined_df)
combined_df = combined_df.drop(columns=['timestamp'], errors='ignore')
combined_df.to_csv('datasets/cleaned_combined_dataset.csv', index=False)

# Write column headings to a text file with spacing
with open('misc/column_headings.txt', 'w') as f:
    for column in combined_df.columns:
        f.write(f"{column}\n\n")
