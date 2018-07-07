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
import calendar
from itertools import *
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

def savePlots(loc, plt):
    plt.savefig(loc)

event_colors = {'CommitCommentEvent':'#e59400',
                'CreateEvent':'#B2912F',
                'DeleteEvent':'#B276B2',
                'ForkEvent':'#4D4D4D', 
                'IssueCommentEvent':'#DECF3F',
                'IssuesEvent':'#60BD68',
                'PullRequestEvent':'#5DA5DA',
                'PullRequestReviewCommentEvent':'#D3D3D3',
                'PushEvent':'#F17CB0',
                'WatchEvent':'#F15854'}

def plot_histogram(data,xlabel,ylabel,title, log=False, loc=False):

    sns.set_style('whitegrid')
    sns.set_context('talk')
    
    ##ploting Histogram
    _,bins = np.histogram(data,bins='doane')

    measurement = pd.DataFrame(data)

    measurement.plot(kind='hist',bins=bins,legend=False,cumulative=False,normed=False,log=log)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    
    if loc != False:
        savePlots(loc,plt)
        return
    
    return plt.show()

def plot_line_graph(data,xlabel,ylabel,title,labels="",loc=False):
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
    
    if loc != False:
        savePlots(loc,plt)
        return
    return plt.show()
    
def plot_time_series(data,xlabel,ylabel,title,loc=False):
    
    plt.clf()
    sns.set_style('whitegrid')
    sns.set_context('talk')
    p = data
    plt.plot(p['date'],p['value'])

  
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20)
    plt.xticks(rotation=45)

    plt.tight_layout()
    
    if loc != False:
        savePlots(loc,plt)
        return
    
    return plt.show()

def plot_contributions_oneline(data,xlabel,ylabel,title,loc=False):

    sns.set_style('whitegrid')
    sns.set_context('talk')

    p = data
    ax = plt.gca()
    labels = [str(x) for x in p.date.values]
    plt.clf()
    plt.plot(p.date.values, p.value.values, label='Unique Users per Day')

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    
    if loc != False:
        savePlots(loc,plt)
        return
        
    return plt.show()

def plot_contributions_twolines(containsDup,noDups,xlabel,ylabel,title,loc=False):

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
    
    if loc != False:
        savePlots(loc,plt)
        return 
    
    return plt.show()

def plot_palma_gini(data,xlabel,ylabel,title,loc=False):
    data.plot(x = 'cum_nodes',y='cum_value',legend=False)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.plot([0,1],[0,1],linestyle='--',color='k')
    plt.tight_layout()
    plt.title(title)
    if loc != False:
        savePlots(loc,plt)
        return
    return plt.show()

def plot_distribution_of_events(data,weekday,loc=False):
    p = pd.DataFrame(data)
    p = p.reset_index()
    if weekday == True:
        p = p.rename(index=str, columns={'weekday': 'date'})
    p = p.reset_index()
    p = p.pivot(index='date', columns='event', values='value').fillna(0)
    tp = p.reset_index()
    tp.set_index('date')
    del tp['date']
    total = tp.sum(axis=1)
    for ele in tp.columns:
        if ele == 'date':
            continue
        tp[ele] = tp[ele]
    
    plt.clf()
    sns.set_style('whitegrid')
    sns.set_context('talk')

    ax = plt.gca()

    calIndex = list(calendar.day_name)
    labels = [str(x)[:10] for x in p.index.values]

    title = 'Days'
    if weekday == True:
        labels = [calIndex[i] for i in range(len(labels))]
        title = 'Weekday'
    my_colors = list(islice(cycle([ '#B2912F', '#4D4D4D', '#DECF3F','#60BD68','#5DA5DA','#D3D3D3','#F17CB0','#F15854','#B276B2', '#e59400']), None, len(tp)))

    tp.plot(ax=ax, color=[event_colors.get(x) for x in tp.columns],rot=0)
    ax.xaxis.set_ticks(np.arange(0,len(labels)))
    ax.set_xticklabels(labels=labels, rotation=45)
    plt.legend()
    plt.title('Distribution of Events per ' + title)
    plt.xlabel(title)
    plt.ylabel('Number of Events')

    plt.tight_layout()
    if loc != False:
        savePlots(loc,plt)
        return
    return plt.show()




#############
#User Centric
#############

def plot_top_users(data, xlabel,ylabel,title, log=False,loc=False):
    data = pd.DataFrame(data)

    data.plot(kind='bar',legend=False,log=log)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.title(title)
    if loc != False:
        savePlots(loc,plt)
        return 
    return plt.show()

def plot_activity_timeline(data,xlabel,ylabel,title, log=False,loc=False):
    p = data
    for u in users:
        p[p['user'] == u]['value'].plot(legend=False,logy=False,label=u)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.xticks(rotation=45)
    if loc != False:
        savePlots(loc,plt)
        return 
    return plt.show()

############
#Community
############

def plot_CommunityProportions(p,xlabel,ylabel,title, loc=False):
    data = pd.DataFrame(p)
    ax = data.plot(kind='bar',legend=False)
    ax.set_xticklabels(data.edgeType.values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if loc != False:
        savePlots(loc,plt)
        return
    return plt.show()


def plot_propIssueEvent(p, xlabel, ylabel,title, loc=False):

    plt.clf()
    fig = plt.figure(figsize=(18,15))
    ax = fig.add_subplot(221)
    labels = [str(x)[:10] for x in p.index.values]
    ys = [x for x in range(len(p[p['issueType'] == 'closed']))]

    plt.plot(ys, p[p['issueType'] == 'closed'].counts.values, label='Closed')
    plt.plot(ys, p[p['issueType'] == 'opened'].counts.values, label='Opened')
    plt.plot(ys, p[p['issueType'] == 'reopened'].counts.values, label='ReOpened')
    ax.set_xticklabels(labels=labels, fontsize=20)

    plt.tight_layout()
    plt.xlabel(xlabel,fontsize=20)
    plt.ylabel(ylabel,fontsize=20)
    plt.title(title,fontsize=20)
    plt.legend(bbox_to_anchor=(-.25, .001), loc=2, prop={'size': 15})
    plt.xticks(rotation=45)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    if loc != False:
        savePlots(loc,plt)
        return
    return plt.show()
    
