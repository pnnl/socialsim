import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import Pool
from functools import partial

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

These measurements assume that the data is in the order id,created_at,type,actor.id,repo.id
'''


'''
This method returns the distributon for the diffusion delay.

Question #1

Inputs: DataFrame - Data
        eventType - A specific event to filter data on

Output: A list (array) of deltas in days
'''
def getRepoDiffusionDelay(df,eventType=None):

    #Standardize Time and Sort Dataframe
    df.columns = ['id','time','event','user','repo']

    #Checks for specific event type, uses both Fork amd WatchEvent
    if eventType is not None:
        df = df[df.event == eventType]

    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by='time')


    #Find difference between event time and "creation time" of repo
    #Creation time is first seen event
    creation_day = df['time'].iloc[0]
    df['delta'] = (df['time']-creation_day).fillna(0)
    df = df.iloc[1:]

    delta = df['delta'].values

    #Converts deltas from seconds to days
    delta = delta / (10 ** 9) / 86400

    delta = [int(ele) for ele in delta]
    return delta



'''
This method returns the growth of a repo over time.

Question #2

Input: df - Dataframe of all data for a repo
       cumSum - This is a boolean that indicates if the dataframe should be cumuluative over time.

       output - A dataframe that describes the repo growth. Indexed on time.
'''
def getRepoGrowth(df, cumSum=True):

    df.columns = ['id', 'time', 'event','user', 'repo']
    df['id'] = df.index
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by='time')

    df.set_index('time', inplace=True)
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year

    p = df[['year', 'month', 'day', 'id']].groupby(['year', 'month', 'day']).count()
    p = pd.DataFrame(p).reset_index()

    if cumSum == True:
        p['id'] = p.cumsum(axis=0)['id']

    p.column = ['year', 'month', 'day', 'count']

    p['date'] = p.apply(lambda x: datetime.strptime("{0} {1} {2}".format(x['year'], x['month'], x['day']), "%Y %m %d"), axis=1)
    p['date'] = p['date'].dt.strftime('%Y-%m-%d')
    p = p.set_index(p['date'])
    del p['year']
    del p['month']
    del p['day']
    del p['date']

    return p


'''
This method returns the the number of events on a repo before it "dies" (deleted or no activity)

Question #2

Input - Dataframe of a repo
Output - Number of events before death
'''
def getLifetimeDepth(df):
    df.columns = ['id', 'time','event','user','repo']
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by='time')
    return len(df)


'''
Time from creation to "death" of repo (deleted or no activity)

Question #2

Input - Dataframe of a repo
Output - Time from creation to "death"
'''
def getLifetimeTime(df):

    df.columns = ['id', 'time', 'event', 'user', 'repo']
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
       dropDup - Boolean to indicate whether or not the measurement should contain duplicate users (on a daily basis)
       cumaltive - Boolean to indicate whether or not the measurement should be cumulative over time

'''
def getContributions(df,dropDup=True,cumulative=True):
    df.columns = ['id', 'time', 'event', 'user', 'repo']
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by='time')

    df.set_index('time', inplace=True)
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year

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
    del p['year']
    del p['month']
    del p['day']

    return p

'''
This method returns the average time between events for each repo

NOTE: Multithreading is highly recommended for datasets with more than 5000 repos.

Question #11

Inputs: df - Data frame of all data for repos
        repos - (Optional) List of specific repos to calculate the measurement for
        nCPu - (Optional) Number of CPU's to run measurement in parallel

Outputs: A list of average times for each repo. Length should match number of repos
'''
def getAvgTimebwEvents(df,repos=None, nCPU=1):
    # Standardize Time and Sort Dataframe
    df.columns = ['id','time', 'event', 'user', 'repo']
    df['time'] = pd.to_datetime(df['time'])

    if repos == None:
        repos = df['repo'].unique()

    pool = Pool(nCPU)
    mean_time_partial = partial(getMeanTime,df.copy())
    deltas = pool.map(mean_time_partial,repos)
    pool.close()
    pool.join()
 
    return deltas
    

'''
Helper function for getting the average time between events

Inputs: Same as average time between events
Output: Same as average time between events
'''
def getMeanTime(df,r):
     d = df[df.repo == r]
     d = d.sort_values(by='time')
     delta = np.mean(np.diff(d.time)) / np.timedelta64(1, 's')
     return delta


'''
This method returns the distribution for each event over time or by weekday. Default is over time.

Question #5

Inputs: df - Data frame of all data for repos
        nCPu - (Optional) Number of CPU's to run measurement in parallel
        weekday - (Optional) Boolean to indicate whether the distribution should be done by weekday. Default is False.

Output: Dataframe with the distribution of events by weekday. Columns: Event, Weekday, Count
'''
def getDistributionOfEvents(df,nCPU = 1,weekday=False):
    df.columns = ['id','time','event','user','repo']
    df_split = np.array_split(df,nCPU)
    pool = Pool(nCPU)
    distribution_partial = partial(processDistOfEvents, weekday=weekday)
    df_list = pool.map(distribution_partial,df_split)
    #Merge List into a single dataframe
    df_1 = df_list[0]
    for i in range(1,len(df_list)):
        df_1 = pd.merge(df_1,df_list[i],on=['event','weekday'],how='left')
    df_1.set_index(['event','weekday'])
    pool.close()
    pool.join()
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
        return p

    else:
        p = df[['event', 'year', 'month', 'id']].groupby(['event', 'year', 'month']).count()
        p = pd.DataFrame(p).reset_index()
        p.column = ['event', 'year', 'month', 'count']
        p['date'] = p.apply(lambda x: datetime.strptime("{0} {1}".format(x['year'], x['month']), "%Y %m"), axis=1)
        p['date'] = p['date'].dt.strftime('%Y-%m')
        return p

'''
This method returns the distribution of event type per repo e.g. x repos with y number of events, z repos with n
 amounts of events.

Question #12,13,16

Inputs: df - Data frame with data for all repos
        eventType - Event time to get distribution over

Outputs: Dataframe with the distribution of event type per repo. Columns are repo id and the count of that event.
'''
def getDistributionOfEventsByRepo(df,eventType):
    df.columns = ['id','time', 'event', 'user', 'repo']
    df = df[df.event == eventType]
    p = df[['repo', 'id']].groupby(['repo']).count()
    p = p.sort_values(by='id')
    df_p = pd.DataFrame(p['id'].values)
    return df_p


'''
This method returns the distribution of repo life over the dataframe. Repo life is defined from the first time a repo
event is seen or created to when it is deleted or the last event in the dataframe.

Question #11

Inputs: df - Data frame with the data for all repos

Outputs: List of deltas for each repos lifetime.
'''
def getDisributionOverRepoLife(df):
    df.columns = ['id', 'time','event','user', 'repo']
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

    return delta

'''
This method returns the gini coefficient for the data frame.

Question #6,14

Input: df - Data frame containing data can be any subset of data
       type - (Optional) This is the type of gini coefficient. Options: user or repo (case sensitive)

Output: g - gini coefficient
'''
def getGiniCoef(df, type='repo'):
    df.columns = ['id', 'time', 'event' ,'user', 'repo']
    df = df[['repo', 'user']].groupby([type]).count()
    df.columns = ['counts']
    df = df.reset_index()

    values = df['counts'].values
    values = np.sort(np.array(values))

    cdf = np.cumsum(values) / float(np.sum(values))
    percent_nodes = np.arange(len(values)) / float(len(values))

    g = 1 - 2*np.trapz(x=percent_nodes,y=cdf)
    return g


'''
This method returns the palma coefficient along with a data frame showing the disparity.

Question #6,14
Input: df - Data frame containing data can be any subset of data
       type - (Optional) This is the type of palma coefficient. Options: user or repo (case sensitive)

Output: p - Palma Coefficient
        data - data frame that represents the event disparity
'''
def getPalmaCoef(df, type='repo'):
    df.columns = ['id','time', 'event', 'user', 'repo']
    df = df[['repo', 'user']].groupby([type]).count()
    df.columns = ['counts']
    df = df.reset_index()
    values = df['counts'].values
    values = np.sort(np.array(values))
    cdf = np.cumsum(values) / float(np.sum(values))
    percent_nodes = np.arange(1, len(values) + 1) / float(len(values))
    p10 = np.sum(values[percent_nodes >= 0.9])
    p40 = np.sum(values[percent_nodes <= 0.4])
    p = float(p10) / float(p40)
    x = cdf
    y = percent_nodes
    data = pd.DataFrame({'cum_nodes': y, 'cum_value': x})

    return p,data