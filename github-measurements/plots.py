import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from datetime import datetime
import seaborn as sns
import matplotlib.dates as dates

def plot_histogram(data,xlabel,ylabel,title, log=False):
    sns.set_style('whitegrid')
    sns.set_context('talk')
    
    ##ploting Histogram
    _,bins = np.histogram(data,bins='auto')
 
    measurement = pd.DataFrame(data)

    measurement.plot(kind='hist',bins=bins,legend=False,cumulative=False,normed=False,log=log)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    
    return plt

def plot_line_graph(data,xlabel,ylabel,title,labels=""):
    sns.set_style('whitegrid')
    sns.set_context('talk')
    
    ##plotting line graph
    _,bins = np.histogram(data,bins='auto')
 
    Watchmeasurement = pd.DataFrame(data)

    tx = [x for x in range(len(data))]
    
    plt.figure(figsize=(10,7))
    plt.plot(tx, data, label=labels)
    
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=15)
    plt.tight_layout()
    return plt.show()
    
def plot_time_series(data,xlabel,ylabel,title):
    
    sns.set_style('whitegrid')
    sns.set_context('talk')
    p = data
    p.plot(legend=False)
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.xticks(rotation=45)
    return plt.show()

def plot_contributions_oneline(data,xlabel,ylabel,title):
    p = data
    fig = plt.figure(figsize=(18,15))
    ax = fig.add_subplot(221)
    labels = [str(x)[:10] for x in p.date.values]
    ys = [x for x in range(len(p))]
    plt.clf()
    plt.plot(ys, p.user.values, label='Unique Users per Day')
    ax.set_xticklabels(labels=labels, fontsize=20)

    # ax.tick_params(labelsize=15)
    plt.tight_layout()
    plt.xlabel('Time',fontsize=20)
    plt.ylabel('Number of Users',fontsize=20)
    plt.title('Cumulative Number of Contributing Users Over Time',fontsize=20)
    plt.legend(loc=2, prop={'size': 15})
    plt.xticks(rotation=45)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    return plt.show()

def plot_contributions_twolines(containsDup,noDups,xlabel,ylabel,title):

    plt.clf()
    fig = plt.figure(figsize=(18,15))
    ax = fig.add_subplot(221)
    labels = [str(x)[:10] for x in containsDup.date.values]
    ys = [x for x in range(len(containsDup))]

    plt.plot(ys, containsDup.user.values, label='Unique Users per Day')
    plt.plot(ys, noDups.user.values, label='Unique User over All')
    ax.set_xticklabels(labels=labels, fontsize=20)

    # ax.tick_params(labelsize=15)
    plt.tight_layout()
    plt.xlabel('Time',fontsize=20)
    plt.ylabel('Number of Users',fontsize=20)
    plt.title('Cumulative Number of Contributing Users Over Time',fontsize=20)
    plt.legend(loc=2, prop={'size': 15})
    plt.xticks(rotation=45)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    return plt.show()

def plot_palma_gini(data,xlabel,ylabel,title):
    data.plot(x = 'cum_nodes',y='cum_value',legend=False)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.plot([0,1],[0,1],linestyle='--',color='k')
    plt.tight_layout()
    plt.title(title)
    return plt.show()

def plot_distribution_of_events(data,weekday):
    p = pd.DataFrame(data)
    p = p.reset_index()
    if weekday == True:
        p = p.rename(index=str, columns={'weekday': 'date'})
    p = p.reset_index()
    p = p.pivot(index='date', columns='event', values=0).fillna(0)
    tp = p.reset_index()
    tp.set_index('date')
    del tp['date']
    total = tp.sum(axis=1)
    for ele in tp.columns:
        if ele == 'date':
            continue
        tp[ele] = tp[ele]
    
    import calendar
    from itertools import *
    plt.clf()
    sns.set_style('whitegrid')
    sns.set_context('talk')

    fig = plt.figure(figsize=(50,25))
    ax = fig.add_subplot(223)

    calIndex = list(calendar.day_name)
    labels = [str(x)[:10] for x in p.index.values]
    print len(labels)
    print calIndex
    title = 'Month'
    if weekday == True:
        labels = [calIndex[i] for i in range(len(labels))]
        title = 'Weekday'
    my_colors = list(islice(cycle([ '#B2912F', '#4D4D4D', '#DECF3F','#60BD68','#5DA5DA','#D3D3D3','#F17CB0','#F15854','#B276B2', '#e59400']), None, len(df)))

    tp.plot(ax=ax, color=my_colors,rot=0)
    ax.xaxis.set_ticks(np.arange(0,len(labels)))
    ax.set_xticklabels(labels=labels, rotation=45)
    ax.tick_params(labelsize=30)
    plt.legend(bbox_to_anchor=(1, 1), prop={'size': 30})
    plt.title('Distribution of Events per ' + title , fontsize=40)
    plt.xlabel(title, fontsize=40)
    plt.ylabel('Number of Events', fontsize=40)

    plt.tight_layout()
    
    return plt.show()




#############
#User Centric
#############

def plot_top_users(data, xlabel,ylabel,title, log=False):
    data = pd.DataFrame(data)

    data.plot(kind='bar',legend=False,log=log)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.title(title)
    return plt.show()

def plot_activity_timeline(data,xlabel,ylabel,title, log=False):
    p = data
    for u in users:
        print p.columns
        p[p['user'] == u]['value'].plot(legend=False,logy=False,label=u)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.xticks(rotation=45)
    return plt.show()
    
