import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_cat_freq(column, title='Freq', ax=None):
    value_counts = column.value_counts()
    if ax is None:
        fig, ax = plt.subplots()

    # Plot the bar chart
    ax.bar(value_counts.index, value_counts.values)

    # Add labels and title
    ax.set_ylabel('Frequency')
    ax.set_title(title)

def plot_all_cat_freq(df, figsize=(15, 15)):
    num_cols = df.select_dtypes(include='object').shape[1]
    num_rows = (num_cols - 1) // 3 + 1  # Calculate the number of rows based on 3 columns per row
    fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=figsize)

    for i, column in enumerate(df.select_dtypes(include='object')):
        row_idx = i // 3
        col_idx = i % 3

        # Check if the column has more than 5 categories
        if len(df[column].unique()) > 5:
            # Select the 5 most frequent categories
            top_5_categories = df[column].value_counts().nlargest(5).index
            plot_cat_freq(df[df[column].isin(top_5_categories)][column].reset_index(drop = True), title=column, ax=axes[row_idx, col_idx])
        else:
            plot_cat_freq(df[column], title=column, ax=axes[row_idx, col_idx])

    plt.tight_layout()
    plt.show()

def remove_outliers(df):
    df_out = df.copy()
    
    # Calculate the first quartile (Q1) and third quartile (Q3)
    Q1 = df_out.quantile(0.25)
    Q3 = df_out.quantile(0.75)
        
    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1
        
    # Calculate the lower and upper outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    for col in df_out.columns:
        df_out = df_out[(df_out[col] >= lower_bound[col]) & (df_out[col] <= upper_bound[col])]
    
    return list(df_out.index)