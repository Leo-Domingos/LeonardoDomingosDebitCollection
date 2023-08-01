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
            plot_cat_freq(df[df[column].isin(top_5_categories)][column], title=column, ax=axes[row_idx, col_idx])
        else:
            plot_cat_freq(df[column], title=column, ax=axes[row_idx, col_idx])

    plt.tight_layout()
    plt.show()

def remove_outliers(df, bound_const = 1.5):
    df_out = df.copy()
    
    # Calculate the first quartile (Q1) and third quartile (Q3)
    Q1 = df_out.quantile(0.25)
    Q3 = df_out.quantile(0.75)
        
    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1
        
    # Calculate the lower and upper outlier boundaries
    lower_bound = Q1 - bound_const * IQR
    upper_bound = Q3 + bound_const * IQR

    for col in df_out.columns:
        df_out = df_out[(df_out[col] >= lower_bound[col]) & (df_out[col] <= upper_bound[col])]
    
    return list(df_out.index)

def create_stacked_bar_plot(ax, df, col_x, col_y, title):
    counts = df.groupby([col_x, col_y]).size().unstack(fill_value=0)
    x_labels = df[col_x].unique()
    sorted_counts = counts.loc[x_labels]
    
    ax = sorted_counts.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(col_x)
    ax.set_ylabel("Contagem")

    # Custom legend with all categories
    handles, labels = ax.get_legend_handles_labels()
    labels = [f"{value:.0f}" for label in labels for value in sorted_counts.columns]
    ax.legend(handles, labels, title=col_y)

    for container in ax.containers:
        ax.bar_label(container, label_type='center')