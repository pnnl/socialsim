import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import Pool
from functools import partial
from plots import *

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


'''
Notice: This computer software was prepared by Battelle Memorial Institute, hereinafter the Contractor, under Contract
No. DE-AC05-76RL01830 with the Department of Energy (DOE).  All rights in the computer software are reserved by DOE on
behalf of the United States Government and the Contractor as provided in the Contract.  You are authorized to use this
computer software for Governmental purposes but it is not to be released or distributed to the public.  NEITHER THE
GOVERNMENT NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS
SOFTWARE.  This notice including this sentence must appear on any copies of this computer software.
'''

'''
This class implements repo centric methods.

These metrics assume that the data is in the order id,created_at,type,actor.id,repo.id
'''


'''
This method returns the distributon for the diffusion delay.

Question #1

Inputs: DataFrame - Data
        eventType - A list of events to filter data on
        unit - Time unit for time differences, e.g. "s","d","h"
        metadata_file - CSV file with repo creation times.  Otherwise use first repo observation as proxy for creation time.

Output: A list (array) of deltas in days
'''

def getRepoDiffusionDelay(df,eventType=None,unit='h',metadata_file = '', plot=False, saveData=False):

    if metadata_file != '':
        repo_metadata = pd.read_csv(metadata_file)
        repo_metadata = repo_metadata[['full_name_h','created_at']]
        repo_metadata['created_at'] = pd.to_datetime(repo_metadata['created_at'])

    #Standardize Time and Sort Dataframe
    df.columns = ['time','event','user','repo']

    #Checks for specific event type, uses both Fork and WatchEvent
    if eventType is not None:
        df = df[df.event.isin(eventType)]

    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by='time')

    if metadata_file != '':
        df = df.merge(repo_metadata,left_on='repo',right_on='full_name_h',how='left')
        df = df[['repo','created_at','time']].dropna()
        df['delta'] = (df['time']-df['created_at']).apply(lambda x: int(x / np.timedelta64(1, unit)))

    else:
        #Find difference between event time and "creation time" of repo
        #Creation time is first seen event
        creation_day = df['time'].min()
        df['delta'] = (df['time']-creation_day).apply(lambda x: int(x / np.timedelta64(1, unit)))
        df = df.iloc[1:]

    delta = df['delta'].values

    if plot==False:
        return delta


    ##############
    ## Plotting ##
    ##############

    if eventType is not None:
        eventList = []
        for ele in eventType:
            eventList.append(ele[:-5])
        eventType = '/'.join(eventList)
        
    else:
        eventType = 'All'

    
    unit_labels = {'s':'Seconds',
                   'h':'Hours',
                   'd':'Days'}

    ##To Save or not
    if saveData != False:
        plot_histogram(delta,unit_labels[unit] + ' Between '+eventType+' Event and Creation Event','Number of Events','Diffusion Delay',loc=saveData + '_histogram.png')

        ##plotting line graph
        plot_line_graph(delta,'Event Number','Delta between '+eventType+' Event and Creation','Diffusion Delay',labels=eventType,loc=saveData + '_linegraph.png')

    else:
        print(plot_histogram(delta,unit_labels[unit] + ' Between '+eventType+' Event and Creation Event','Number of Events','Diffusion Delay',loc=saveData))
        

        ##plotting line graph
        print(plot_line_graph(delta,'Event Number','Delta between '+eventType+' Event and Creation','Diffusion Delay',labels=eventType,loc=saveData))
        

    return delta


'''
This method returns the growth of a repo over time.

Question #2

Input: df - Dataframe of all data for a repo
       cumSum - This is a boolean that indicates if the dataframe should be cumuluative over time.

       output - A dataframe that describes the repo growth. Indexed on time.
'''

def getRepoGrowth(df, cumSum=False, plot=False, saveData=False):
    df.columns = ['time', 'event','user', 'repo']
    df['id'] = df.index
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by='time')

    df.set_index('time', inplace=True)
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year

    #get daily event counts
    p = df[['year', 'month', 'day', 'id']].groupby(['year', 'month', 'day']).count()
    p = pd.DataFrame(p).reset_index()

    #get cumulative sum of daily event counts
    if cumSum == True:
        p['id'] = p.cumsum(axis=0)['id']

    p.columns = ['year', 'month', 'day', 'value']

    p['date'] = p.apply(lambda x: datetime.strptime("{0} {1} {2}".format(x['year'], x['month'], x['day']), "%Y %m %d"), axis=1)
    p['date'] = pd.to_datetime(p['date'].dt.strftime('%Y-%m-%d'))
    p = p.set_index(p['date'])
    del p['year']
    del p['month']
    del p['day']
    del p['date']
    p = p.reset_index()
    if plot== False:
        return p

    ##############
    ## Plotting ##
    ##############

    cumTitle = ''
    if cumSum:
        cumTitle = 'Cumulative Sum of '

    if saveData != False:
        plot_time_series(p,'Time','Total Number of Events', cumTitle + 'Events Over Time', loc=saveData+'_time_series_cumsum'+str(cumSum)+'.png')

        #To mimic PNNL Graph, run with CumSum as False
        plot_histogram(p['value'].values,'Events Per Day',cumTitle + 'Total Number of Days','Distribution Over Daily Event Counts', loc=saveData + 'histogram_cumsum' +str(cumSum)+'.png')

    else:
        print(plot_time_series(p,'Time','Total Number of Events',cumTitle + 'Events Over Time'))
        

        #To mimic PNNL Graph, run with CumSum as False
        print(plot_histogram(p['value'].values,'Events Per Day',cumTitle + 'Total Number of Days','Distribution Over Daily Event Counts'))
        

    return p

'''
This method returns the the number of events on a repo before it "dies" (deleted or no activity)

Question #2

Input - Dataframe of a repo
Output - Number of events before death
'''
def getLifetimeDepth(df):
    df.columns = ['time','event','user','repo']
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by='time')
    return len(df)

'''
Time from creation to "death" of repo (deleted or no activity)

Question #2

Input - Dataframe of a repo
Output - Time from creation to "death" (default is days)
'''
def getLifetimeTime(df):

    df.columns = ['time', 'event', 'user', 'repo']
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by='time')
    p = pd.DataFrame(df.iloc[[0, -1]])
    p['delta'] = (p['time'] - p['time'].shift()).fillna(0)
    p['ans'] = p['delta'].apply(lambda x: x.total_seconds()).astype('int64') / (24 * 60 * 60)
    return p['ans'].values[1]


'''
Calcultes the number of contributers, with or without duplicates.

Question # 4

Input: df - Data frame can be repo or non-repo centric
       dropDup - Boolean to indicate whether or not the metric should contain duplicate users (on a daily basis), if None run Both
       cumaltive - Boolean to indicate whether or not the metric should be cumulative over time

'''
def getContributions(df,dropDup=False,cumulative=False, plot=False, saveData=False, wfEvents=True):
    def contributionsHelper(df,dropDup,cumulative):
        if dropDup:
            df = df.drop_duplicates(subset=['user'])

        p = df[['user', 'year', 'month', 'day']].groupby(['year', 'month', 'day']).nunique()
        p = pd.DataFrame(p)
        del p['day']
        del p['year']
        del p['month']
        p = p.reset_index()

        if cumulative:
            p['user'] = p.cumsum(axis=0)['user']

        p['date'] = p.apply(lambda x: datetime.strptime("{0} {1} {2}".format(x['year'], x['month'], x['day']), "%Y %m %d"),
                        axis=1)
        p['date'] = p['date'].dt.strftime('%Y-%m-%d')
        p['date'] = pd.to_datetime(p['date'])
        del p['year']
        del p['month']
        del p['day']
        return p

    df.columns = [ 'time', 'event', 'user', 'repo']
    df['time'] = pd.to_datetime(df['time'])

    if wfEvents == True:
        df = df[(df.event != 'ForkEvent') & (df.event != 'WatchEvent')]

    df = df.sort_values(by='time')

    df.set_index('time', inplace=True)
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year


    #Running Both Duplicate and Non Duplicate
    if dropDup == None:
        #run both
        noDups = contributionsHelper(df, True, cumulative)
        containsDup = contributionsHelper(df,False, cumulative)

        noDups = noDups.reset_index(drop=True)
        containsDup = containsDup.reset_index(drop=True)


        ##############
        ## Plotting ##
        ##############

        if saveData != False:
            plot_contributions_twolines(containsDup,noDups,'Time','Number of Users','Number of Contributing Users Over Time', loc=saveData + '_containsDup_contributions_cumulative_' +str(cumulative)+ '.png')
        else:
            print(plot_contributions_twolines(containsDup,noDups,'Time','Number of Users','Cumulative Number of Contributing Users Over Time'))
            
        return noDups, containsDup

    else:

        
        results = contributionsHelper(df,dropDup, cumulative)

        ##############
        ## Plotting ##
        ##############
        title = 'with'
        if dropDup:
            title = 'without'

        cumTitle = ''
        if cumulative:
            cumTitle = 'Cumulative '

        p = results
        p = p.rename(columns={"user": "value"})


        if not plot:
            return p

        sns.set_style('whitegrid')
        sns.set_context('talk')

        if saveData != False:
            # To mimic PNNl's output have the cumulative as True
            plot_contributions_oneline(p,'Time','Number of Users',cumTitle +'Number of Users Over Time', loc=saveData + '_one_line_no_duplicates_cumsum' + str(cumulative) + '.png')

            #To mimic PNNL's output have cumulative as False
            plot_histogram(p.value.values,'Total Number of Contributors','Days',cumTitle+'Distribution of Unique Contributors '+title+' Duplicates', loc=saveData + '_histogram_no_duplicates_cumsum' + str(cumulative) + '.png')

        else:
            # To mimic PNNl's output have the cumulative as True
            #plot_contributions_oneline(p,'Time','Number of Users', cumTitle + 'Number of Contributing Users Over Time')

            #To mimic PNNL's output have cumulative as False
            #plot_histogram(p.user.values,'Total Number of Contributors','Days', cumTitle + 'Distribution of Unique Contributors '+title+' Duplicates')
            pass

        return p

'''
This method returns the average time between events for each repo

NOTE: Multithreading is highly recommended for datasets with more than 5000 repos.

Question #12

Inputs: df - Data frame of all data for repos
        repos - (Optional) List of specific repos to calculate the metric for
        nCPu - (Optional) Number of CPU's to run metric in parallel

Outputs: A list of average times for each repo. Length should match number of repos. Elements with NaN correspond to a
repo only having a single event.
'''
def getAvgTimebwEvents(df,repos=None, nCPU=1, plot=False, saveData=False):
    # Standardize Time and Sort Dataframe
    df.columns = ['time', 'event', 'user', 'repo']
    df['time'] = pd.to_datetime(df['time'])

    if repos == None:
        repos = df['repo'].unique()

    p = Pool(nCPU)
    args = [(df, repos[i]) for i, item_a in enumerate(repos)]
    deltas = p.map(getMeanTimeHelper,args)
    p.join()
    p.close()

    if plot==False:
        return deltas

    if saveData != False:
        plot_histogram(deltas,'Time Between PullRequestEvents in Seconds','Number of Repos','Average Time Between PullRequestEvents for ' + community, loc=saveData + '_histogram.png')
    else:
        print(plot_histogram(deltas,'Time Between PullRequestEvents in Seconds','Number of Repos','Average Time Between PullRequestEvents for' + community))
        

'''
Helper function for getting the average time between events

Inputs: Same as average time between events
Output: Same as average time between events
'''
def getMeanTime(df, r):
    d = df[df.repo == r]
    d = d.sort_values(by='time')
    delta = np.mean(np.diff(d.time)) / np.timedelta64(1, 's')
    return delta


def getMeanTimeHelper(args):
    return getMeanTime(*args)
    return deltas


'''
This method returns the distribution for each event over time or by weekday. Default is over time.

Question #5

Inputs: df - Data frame of all data for repos
        nCPu - (Optional) Number of CPU's to run metric in parallel
        weekday - (Optional) Boolean to indicate whether the distribution should be done by weekday. Default is False.

Output: Dataframe with the distribution of events by weekday. Columns: Event, Weekday, Count
'''
def getDistributionOfEvents(df,nCPU = 1,weekday=False, plot=False, saveData=False):
    df.columns = ['time','event','user','repo']
    df['id'] = df.index
    df_split = np.array_split(df,nCPU)
    pool = Pool(nCPU)
    distribution_partial = partial(processDistOfEvents, weekday=weekday)
    df_list = pool.map(distribution_partial,df_split)
    pool.close()
    pool.join()
    # Merge List into a single dataframe
    sum_col = "user" if weekday else "id"
    if weekday:
        columns = ['event', 'weekday']
    else:
        columns = ['event', 'date']
    df_1 = df_list[0]
    for i in range(1, len(df_list)):
        df_1 = pd.merge(df_1, df_list[i], on=columns, how='outer')
    df_1 = df_1[columns + ['value']]
    if plot==False:
        return df_1

    ##############
    ## Plotting ##
    ##############
    if saveData != False:
        plot_distribution_of_events(df_1,weekday, loc=saveData + '.png')
    else:
        print(plot_distribution_of_events(df_1,weekday))
        

    return df_1

'''
Helper Function for getting the Dist. of Events per weekday.
'''
def processDistOfEvents(df,weekday):

    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by='time')
    df.set_index('time', inplace=True)
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year

    if weekday:
        df['weekday'] =df.apply(lambda x:datetime(x['year'],x['month'],x['day']).weekday(),axis=1)
        p = df[['event','user','weekday']].groupby(['event','weekday']).count()
        p.columns = ['value']
        p = p.reset_index()
        return p

    else:
        p = df[['event', 'year', 'month', 'day','id']].groupby(['event', 'year', 'month','day']).count()
        p = pd.DataFrame(p).reset_index()
        p.columns = ['event', 'year', 'month','day','value']
        p['date'] = p.apply(lambda x: datetime.strptime("{0} {1} {2}".format(x['year'], x['month'],x['day']), "%Y %m %d"), axis=1)
        p['date'] = p['date'].dt.strftime('%Y-%m-%d')
        p = p.reset_index()
        return p


'''
This method returns the distribution of event type per repo e.g. x repos with y number of events, z repos with n
 amounts of events.

Question #12,13,14

Inputs: df - Data frame with data for all repos
        eventType - Event time to get distribution over

Outputs: Dataframe with the distribution of event type per repo. Columns are repo id and the count of that event.
'''
def getDistributionOfEventsByRepo(df,eventType='WatchEvent', plot=False, saveData=False,log=True):
    df.columns = ['time', 'event', 'user', 'repo']
    df = df[df.event == eventType]
    p = df[['repo', 'event']].groupby(['repo']).count()
    p = p.sort_values(by='event')
    p.columns = ['value']
    p = p.reset_index()
    if plot == False:
        return p

    ##############
    ## Plotting ##
    ##############

    ylabel = 'Number of Repos '
    if log == True:
        ylabel += '(Log)'

    if saveData != False:
        plot_histogram(p['value'].values,'Number of Events',ylabel,'Distribution of '+eventType+' across Repos',log=log, loc=saveData + '_histogram.png')
    else:
        print(plot_histogram(p['value'].values,'Number of Events',ylabel,'Distribution of '+eventType+' across Repos',log=log))
        

    return p


'''
This method returns the top-k repos by event count for a certain event type

Question #12,13

Inputs: df - Data frame with data for all repos
        eventType - Event time to get distribution over

Outputs: Dataframe with the top-k repos and their event counts. Columns are repo id and the count of that event.
'''
def getTopKRepos(df,k=100,eventType='WatchEvent',plot=False,saveData=False):
    df.columns = ['time', 'event', 'user', 'repo']
    df = df[df.event == eventType]
    p = df[['repo', 'event']].groupby(['repo']).count()
    p = p.sort_values(by='event',ascending=False)
    p.columns = ['value']
    return p.head(k)


'''
This method returns the distribution of repo life over the dataframe. Repo life is defined from the first time a repo
event is seen or created to when it is deleted or the last event in the dataframe.

Question #12

Inputs: df - Data frame with the data for all repos

Outputs: List of deltas for each repos lifetime.
'''
def getDisributionOverRepoLife(df, plot=False, log = True, saveData=False):
    df.columns = ['time','event','user', 'repo']
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by='time')
    df_max = df[['repo', 'time']].groupby('repo').max()
    df_min = df[['repo', 'time']].groupby('repo').min()

    df_min = df_min.reset_index()
    df_max = df_max.reset_index()
    df_min.columns = ['repo', 'minTime']
    m = df_min.merge(df_max)
    m['delta'] = m[['time']].sub(m['minTime'], axis=0)

    delta = m['delta'].values

    #Converts deltas to days (modify as necessary)
    delta = delta / (10 ** 9) / 86400
    delta = [int(x) for x in delta if int(x) <= 25]

    if plot == False:
        return delta

    ##############
    ## Plotting ##
    ##############

    if saveData != False:
        plot_histogram(delta,'Length of Repo Life in Days','Number of Repos (Log)','Distrbution of Repo Life',log=log, loc=saveData + '_histogram.png')
    else:
        print(plot_histogram(delta,'Length of Repo Life in Days','Number of Repos (Log)','Distrbution of Repo Life',log=log))
        

    return delta

'''
This method returns the gini coefficient for the data frame.

Question #6,15

Input: df - Data frame containing data can be any subset of data
       type - (Optional) This is the type of gini coefficient. Options: user or repo (case sensitive)

Output: g - gini coefficient
'''
def getGiniCoef(df, type='repo', plot=False, saveData=False):
    df.columns = ['time', 'event' ,'user', 'repo']
    df = df[['repo', 'user']].groupby([type]).count()
    df.columns = ['counts']
    df = df.reset_index()

    values = df['counts'].values
    values = np.sort(np.array(values))

    cdf = np.cumsum(values) / float(np.sum(values))
    percent_nodes = np.arange(len(values)) / float(len(values))

    g = 1 - 2*np.trapz(x=percent_nodes,y=cdf)

    if plot == False:
        return g

    x = cdf
    y = percent_nodes
    data = pd.DataFrame({'cum_nodes': y, 'cum_value': x})
    ##############
    ## Plotting ##
    ##############

    if saveData != False:
        plot_palma_gini(data,'Cumulative share of Repos','Cumulative share of Events','Repos Event Disparity', loc=saveData+'.png')
    else:
        print(plot_palma_gini(data,'Cumulative share of Repos','Cumulative share of Events','Repos Event Disparity'))
        

    return g


'''
This method returns the palma coefficient along with a data frame showing the disparity.

Question #6,15,33

Input: df - Data frame containing data can be any subset of data
       type - (Optional) This is the type of palma coefficient. Options: user or repo (case sensitive)

Output: p - Palma Coefficient
        data - data frame that represents the event disparity
'''
def getPalmaCoef(df, type='repo', plot=False, saveData=False):
    df.columns = ['time', 'event', 'user', 'repo']
    df = df[['repo', 'user']].groupby([type]).count()
    df.columns = ['counts']
    df = df.reset_index()
    values = df['counts'].values
    values = np.sort(np.array(values))
    cdf = np.cumsum(values) / float(np.sum(values))
    percent_nodes = np.arange(1, len(values) + 1) / float(len(values))
    p10 = np.sum(values[percent_nodes >= 0.9])
    p40 = np.sum(values[percent_nodes <= 0.4])
    try:
    	p = float(p10) / float(p40)
    except ZeroDivisionError:
        p = None
    x = cdf
    y = percent_nodes
    data = pd.DataFrame({'cum_nodes': y, 'cum_value': x})

    if plot == False:
        return p

    #############
    ## Plotting ##
    #############

    if saveData != False:
        plot_palma_gini(data,'Cumulative share of Repos','Cumulative share of Events','Repos Event Disparity', loc=saveData + '.png')
    else:
        print(plot_palma_gini(data,'Cumulative share of Repos','Cumulative share of Events','Repos Event Disparity'))
        

    return p
