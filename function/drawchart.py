#%% Import Lib
from datetime import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%% Config
plt.rcParams['figure.figsize'] = (16,12)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 16
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#%%
def plot_bar_chart(column_name, data, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    sns_countplot = sns.countplot(x=column_name, data=data, palette=sns.color_palette("BuGn"))
    plt.title(f'Label of {column_name}')
    plt.ylabel('Count')

    for patch in sns_countplot.patches:
        height = patch.get_height()
        sns_countplot.annotate(f'{height}', (patch.get_x() + patch.get_width() / 2., height), ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    plt.show()

def plot_histogram_chart(column_name, data):
    plt.figure(figsize=(12,6))
    color = sns.color_palette("BuGn").as_hex()[2]
    plt.hist(data[column_name], bins='auto', color=color, edgecolor='k')
    plt.title(f'Histogram of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.show()

def scatter_plot(x, y, title='', xlabel='', ylabel='', title_fontsize=20, xlabel_fontsize=14, ylabel_fontsize=14,
                 figsize=(10, 6)):
    color = sns.color_palette("BuGn").as_hex()[2]

    plt.figure(figsize=figsize)
    plt.scatter(x, y, color=color, alpha=0.5)
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    plt.show()

def plot_actual_vs_predicted(X_test, y_test, y_pred, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test.index, y_test, label='Actual', alpha=0.5, color='blue')
    plt.scatter(X_test.index, y_pred, label='Predicted', alpha=0.5, color='green')
    plt.title(f'Actual vs. Predicted {title}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


