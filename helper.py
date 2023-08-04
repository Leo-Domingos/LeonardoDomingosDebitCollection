import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.metrics import precision_score
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns

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

class Pipeline():
    def __init__(
            self, 
            df, 
            categorical, 
            numerical, 
            target, 
            TRAIN_RATE = 0.7, 
            TEST_RATE = 0.2, 
            VAL_RATE = 0.1,
            R_SEED = 42,
            label_encoder = LabelEncoder(),
            cat_encoder = OneHotEncoder(sparse_output=False),
            num_normalizer = MinMaxScaler()
            ):
        self.categorical = categorical
        self.numerical = numerical
        self.target = target
        self.TRAIN_RATE = TRAIN_RATE
        self.TEST_RATE = TEST_RATE
        self.VAL_RATE = VAL_RATE
        self.R_SEED = R_SEED
        self.label_encoder = label_encoder
        self.cat_encoder = cat_encoder
        self.num_normalizer = num_normalizer
        self.datetime = 'disconnection_date'
        self.df_n = df.loc[:,self.categorical+self.numerical+[self.target, self.datetime]]

    def split_df(self):
        self.df_val = self.df_n.nlargest(int(self.df_n.shape[0]*self.VAL_RATE), self.datetime)
        aux = self.df_n.loc[~self.df_n.index.isin(self.df_val.index)]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            aux.drop([self.target, self.datetime], axis = 1),
            aux[[self.target]], 
            test_size=self.TEST_RATE/(1-self.VAL_RATE), 
            random_state=self.R_SEED,
            stratify=aux[[self.target]]
            )
    
    def feature_processing(self, multiclass = False):
        self.cat_encoder.fit(self.X_train[self.categorical])
        X_categorical = pd.DataFrame(self.cat_encoder.transform(self.X_train[self.categorical]), columns=self.cat_encoder.get_feature_names_out())
        self.num_normalizer.fit(self.X_train[self.numerical])
        X_numeric_normalized = pd.DataFrame(self.num_normalizer.transform(self.X_train[self.numerical]), columns=self.numerical)
        self.X_processed = pd.concat([X_categorical, X_numeric_normalized], axis=1)
        
        X_categorical_test = pd.DataFrame(self.cat_encoder.transform(self.X_test[self.categorical]), columns=self.cat_encoder.get_feature_names_out())
        X_numeric_normalized_test = pd.DataFrame(self.num_normalizer.transform(self.X_test[self.numerical]), columns=self.numerical)
        self.X_processed_test = pd.concat([X_categorical_test, X_numeric_normalized_test], axis=1)

        if multiclass:
            self.label_encoder.fit(self.y_train.values.ravel())
            self.y_processed = self.label_encoder.transform(self.y_train.values.ravel())
            self.y_processed_test = self.label_encoder.transform(self.y_test.values.ravel())
        else:
            self.y_processed = self.y_train
            self.y_processed_test = self.y_test

    def preprocessing(self, multiclass = False):
        self.split_df()
        self.feature_processing(multiclass)

def cv(X, y, model, num_folds=5, metric = precision_score):
    fold_size = len(X) // num_folds
    scores = []
    
    for i in range(num_folds):
        start, end = i * fold_size, (i + 1) * fold_size
        
        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)
        
        X_valid = X[start:end]
        y_valid = y[start:end]

        model.fit(X_train, y_train.ravel())

        y_pred = model.predict(X_valid)
        score = metric(y_valid, y_pred)
        scores.append(score)
        
    return scores

def select_best_categorical_features(df, target_column):
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    y_encoded = df_encoded[target_column]
    X_encoded = df_encoded.drop(columns=[target_column])
    
    p_values = {}
    for col in X_encoded.columns:
        contingency_table = pd.crosstab(X_encoded[col], y_encoded)
        chi2, p, _, _ = chi2_contingency(contingency_table)
        p_values[col] = p

    sorted_categorical_columns = sorted(p_values, key=p_values.get)

    print("Categorical columns ranked by p-values:")
    for col in sorted_categorical_columns:
        print(f"{col}: p-value = {p_values[col]:.4f}")

    return sorted_categorical_columns

# Function to plot correlation with binary variable
def plot_binary_correlation(df, binary_var):
    categorical_vars = [col for col in df.columns if col != binary_var]

    # Initialize a grid of plots
    n_rows = len(categorical_vars)
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 4 * n_rows))

    # Plot correlation for each categorical variable
    for i, cat_var in enumerate(categorical_vars):
        sns.barplot(data=df, x=cat_var, y=binary_var, ax=axes[i])
        axes[i].set_ylabel('Average ' + binary_var)
        axes[i].set_title('Correlation between ' + binary_var + ' and ' + cat_var)

    plt.tight_layout()
    plt.show()