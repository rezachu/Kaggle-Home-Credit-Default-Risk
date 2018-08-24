# Import Library
import numpy as np
import time
import os
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 18
from tqdm import tqdm_notebook as tqdm
import pandas as pd
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import preprocessing
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns
import datetime as dt
import zipfile
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import cufflinks as cf
cf.go_offline()

pd.set_option('display.max_columns', None)  


# Missing Data Checking
def check_missing(data, head=5):
    data_missing = data.isnull().sum().sort_values(ascending = False)
    percent = (data_missing/len(data))*100
    data_missing_table = pd.concat([data_missing, percent], axis=1, keys=['Count', 'Percentage'])
    return data_missing_table.head(head)
   
# seaborn distribution plot
def sns_dist(data, col, plot_name):
    plt.figure(figsize=(12,5))
    plt.title(plot_name)
    ax = sns.distplot(data[col])
    return

# Plotly Distribution Chart
def plotly_dist(data, col, plot_name):

    temp = data[col].value_counts()
    trace = go.Bar(
        x = temp.index,
        y = (temp / temp.sum())*100,)
    input_ = [trace]
    layout = go.Layout(
        title = plot_name,
        xaxis=dict(
            title="Items of "+ plot_name,
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
        ),
        yaxis=dict(
            title= "Distribution of " + plot_name + "'s target",
            titlefont=dict(
                size=16,
                color='rgb(107, 107, 107)'
            ),
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
    )
    )
    fig = go.Figure(data=input_, layout=layout)
    py.iplot(fig, filename=plot_name + ' Type')

    return 

# Plotly Distribution Chart
def plotly_dist_compare(data, col, plot_name):
    temp = data[col].value_counts()
    #print(temp.values)
    temp_y0 = []
    temp_y1 = []
    for val in temp.index:
        temp_y1.append(np.sum(data["TARGET"][data[col]==val] == 1))
        temp_y0.append(np.sum(data["TARGET"][data[col]==val] == 0))

    trace1 = go.Bar(
        x = temp.index,
        y = (temp_y1 / temp.sum()) * 100,
        name='Difficulties'
    )
    trace2 = go.Bar(
        x = temp.index,
        y = (temp_y0 / temp.sum()) * 100, 
        name='Other Cases'
    )

    data = [trace1, trace2]
    layout = go.Layout(
        title = plot_name,
        #barmode='stack',
        width = 1000,
        xaxis=dict(
            title= "Items of "+ plot_name,
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
        ),
        yaxis=dict(
            title="Distribution of " + plot_name + "'s target in %",
            titlefont=dict(
                size=16,
                color='rgb(107, 107, 107)'
            ),
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
    )
    )

    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
    
    return


# Plotly Distribution Chart
def plotly_dist_target_count(data, col, plot_name):
    temp = data[col].value_counts()
    temp_y1 = []
    for val in temp.index:
        temp_y1.append(np.sum(data["TARGET"][data[col]==val] == 1))

    trace = go.Bar(
        x = temp.index,
        y = temp_y1,)
    input_ = [trace]
    layout = go.Layout(
        title = "Distribution of " + plot_name + ' Type',
        xaxis=dict(
            title="Distribution of " + plot_name + ' Type',
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
        ),
        yaxis=dict(
            title= "Distribution of " + plot_name + ' Type in %',
            titlefont=dict(
                size=16,
                color='rgb(107, 107, 107)'
            ),
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
    )
    )
    fig = go.Figure(data=input_, layout=layout)
    py.iplot(fig, filename=plot_name)
    
    return

# Plotly Pie Chart
def plotly_pie(data, col, label=None):
    values = data[col].value_counts()
    if label == None:
        labels = values.index
    else:
        labels = label
    trace = go.Pie(labels=labels, values=values)
    py.iplot([trace], filename='basic_pie_chart')
    
    return


# Basic Statistial Exploration with distribution plot
def stat_explor_col(data, col, q1, q2, plot_name, chart_type="sns"):

    Max = np.max(data[col])
    Min = np.min(data[col])
    Q1 = data[col].quantile(q1)
    Q2 = data[col].quantile(q2)
    IQR = Q2 - Q1
    print ("The max is", Max)
    print ("The min is", Min)
    print ("The {}% quantile is {}".format(q1*100, Q1))
    print ("The {}% quantile is {}".format(q2*100, Q2))
    print ("The inter quantile range is",IQR)

    temp = data[data[col] >= Q1]
    temp = temp[temp[col] <= Q2].reset_index(drop=True)
    
    if chart_type == "sns":
        sns_dist(temp, col, plot_name)
    elif chart_type == "plotly":
        plotly_dist(temp, col, plot_name)
        
    return
    
# PCA with plot
def pca_results(good_data, pca):
    '''
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    Visualizes the PCA results
    '''

    # Dimension indexing
    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = good_data.keys())
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions

    # Create a bar plot visualization
    fig, ax = plt.subplots(figsize = (14,8))

    # Plot the feature weights as a function of the components
    components.plot(ax = ax, kind = 'bar');
    ax.set_ylabel("Feature Weights")
    ax.set_xticklabels(dimensions, rotation=0)


    # Display the explained variance ratios
    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis = 1)

# Normalize Data
def norm_data(df):
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return pd.DataFrame(x_scaled)

# Run PCA
def pca_test(data, n_c):
    pca = PCA(n_components=n_c)
    pca.fit(data)
    
    return pca_results(data, pca)

# Get PCA Reduced Data
def pca_reduce_data(data, n_c):    
    pca = PCA(n_components=n_c)
    pca.fit(data)
    reduced_data = pca.transform(data)
    
    return reduced_data

# Fill NaN with mean
def fillna_mean(data, col, q1, q2):
    Q1 = data[col].quantile(q1)
    Q2 = data[col].quantile(q2)
    result = data[data[col] >= Q1]
    a = result[result[col] <= Q2]
    data[col] = data[col].fillna(np.mean(a[col]))
    print ("Fill NaN for column {} is done".format(col))
    
    return 

# Fill NaN with unknown
def fillna_obj_unknown(data, col):
    data[col] = data[col].fillna("unknown")
    print ("Fill NaN for column {} is done".format(col))
    return 

# Remove Outlier
def remove_out(data, col, q1, q2):
    Q1 = data[col].quantile(q1)
    Q2 = data[col].quantile(q2)
    result = data[data[col] >= Q1]
    result = result[result[col] <= Q2]
    print (len(result))
    print ("Remove for column {} is done".format(col))
    
    return result

def encode(data, Col):
    data[Col] = data[Col].fillna("Unknown")
    le = preprocessing.LabelEncoder()
    enc = data[Col].value_counts().index
    print (Col)
    le.fit(enc)
    
    return le.transform(data[Col])





