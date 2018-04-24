import pandas as pd
import numpy as np
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
This class implements user centric method. Each function will describe which metric it is used for according
to the questions number and mapping.

These metrics assume that the data is in the order id,created_at,type,actor.id,repo.id
'''


'''
This method returns the number of unique repos that a particular set of users contributed too

Question #17

Inputs: DataFrame - Desired dataset
        users - A list of users of interest

Output: A dataframe with the user id and the number of repos contributed to
'''
def getUserUniqueRepos(df,users=None):
    df = df.copy()
    df.columns = ['time', 'event','user', 'repo']
    if users:
        df = df[df.user.isin(users)]
    df =df.groupby('user')
    data = df.repo.nunique().reset_index()
    data.columns = ['user','value']
    return data


'''
This method returns the cumulative activity of the desire user over time.

Question #19

Inputs: DataFrame - Desired dataset
        users - A list of users of interest

Output: A grouped dataframe of the users activity over time
'''
def getUserActivityTimeline(df, users=None,time_bin='1d',cumSum=False):
    df = df.copy()
    df.columns = ['time', 'event','user', 'repo']
    df['time'] = pd.to_datetime(df['time'])
    if users:
        df = df[df.user.isin(users)]
    df['value'] = 1
    if cumSum:
        df['cumsum'] = df.groupby('user').value.transform(pd.Series.cumsum)
        df = df.groupby(['user',pd.Grouper(key='time',freq=time_bin)]).max().reset_index()
        df['value'] = df['cumsum']
        df = df.drop('cumsum',axis=1)
    else:
        df = df.groupby(['user',pd.Grouper(key='time',freq=time_bin)]).sum().reset_index()

    #timeGrouper
    data = df.sort_values(['user', 'time'])
    return data

'''
This method returns the top k most popular users for the dataset, where popularity is measured
as the total popularity of the repos created by the user.

Question #25

Inputs: DataFrame - Desired dataset
        k - (Optional) The number of users that you would like returned.
        use_metadata - External metadata file containing repo owners.  Otherwise use first observed user with a creation event as a proxy for the repo owner.

Output: A dataframe with the user ids and number events for that user
'''
def getUserPopularity(df,k=10,metadata_file = ''):

    if metadata_file != '':
        repo_metadata = pd.read_csv(metadata_file)
        repo_metadata = repo_metadata[['full_name_h','owner.login_h']]

    df = df.copy()
    df.columns = ['time', 'event','user', 'repo']
    df['value'] = 1
    
    repo_popularity = df[df['event'].isin(['ForkEvent','WatchEvent'])].groupby('repo')['value'].sum().reset_index()

    if metadata_file != '':
        merged = repo_popularity.merge(repo_metadata,left_on='repo',right_on='full_name_h',how='left')
    else:
        user_repos = df[df['event'] == 'CreateEvent'].sort_values('time').drop_duplicates(subset='repo',keep='first')
        user_repos = user_repos[['user','repo']]
        user_repos.columns = ['owner.login_h','repo']
        merged = user_repos.merge(repo_popularity,on='repo',how='left')
        
    measurement = merged.groupby('owner.login_h').value.sum().sort_values(ascending=False).head(k)
    measurement = pd.DataFrame(measurement).sort_values('value',ascending=False)
    return measurement

'''
This method returns the average time between events for each user

Question #29b and c

Inputs: df - Data frame of all data for repos
        users - (Optional) List of specific users to calculate the metric for
        nCPu - (Optional) Number of CPU's to run metric in parallel

Outputs: A list of average times for each user. Length should match number of repos
'''
def getAvgTimebwEvents(df,users=None, nCPU=1):
    df = df.copy()
    df.columns = ['time', 'event', 'user', 'repo']
    df['time'] = pd.to_datetime(df['time'])

    if users == None:
        users = df['user'].unique()

    p = Pool(nCPU)
    args = [(df, users[i]) for i, item_a in enumerate(users)]
    deltas = p.map(getMeanTimeHelper, args)
    p.join()
    p.close()
    return deltas

'''
Helper function for getting the average time between events

Inputs: Same as average time between events
Output: Same as average time between events
'''
def getMeanTime(df, user):
    d = df[df.user == user]
    d = d.sort_values(by='time')
    delta = np.mean(np.diff(d.time)) / np.timedelta64(1, 's')
    return delta


def getMeanTimeHelper(args):
    return getMeanTime(*args)

'''
This method returns distribution the diffusion delay for each user

Question #27

Inputs: DataFrame - Desired dataset
        unit - (Optional) This is the unit that you want the distribution in. Check np.timedelta64 documentation
        for the possible options
        metadata_file - File containing user account creation times.  Otherwise use first observed action of user as proxy for account creation time.

Output: A list (array) of deltas in units specified
'''
def getUserDiffusionDelay(df,unit='s',metadata_file = ''):

    if metadata_file != '':
        user_metadata = pd.read_csv(metadata_file)
        user_metadata['created_at'] = pd.to_datetime(user_metadata['created_at'])


    df = df.copy()
    df.columns = ['time','event','user','repo']
    df['value'] = df['time']
    df['value'] = pd.to_datetime(df['value'])

    if metadata_file != '':
        df = df.merge(user_metadata[['login_h','created_at']],left_on='user',right_on='login_h',how='left')
        df = df[['login_h','created_at','value']].dropna()
        measurement = df['value'].sub(df['created_at']).apply(lambda x: int(x / np.timedelta64(1, unit)))
    else:
        grouped = df.groupby('user')
        transformed = grouped['value'].transform('min')
        measurement = df['value'].sub(transformed).apply(lambda x: int(x / np.timedelta64(1, unit)))
    
   

    return measurement


'''
This method returns the gini coefficient for user events. (User Disparity)

Question #26a

Inputs: DataFrame - Desired dataset


Output: The gini coefficient for the dataset
'''
def getGiniCoef(df):
    df = df.copy()
    df.columns = ['time', 'event', 'user', 'repo']
    df['value'] = 1
    df = df.groupby('user')
    event_counts = df.value.sum()
    values = np.sort(np.array(event_counts))

    cdf = np.cumsum(values) / float(np.sum(values))
    percent_nodes = np.arange(len(values)) / float(len(values))

    g = 1 - 2*np.trapz(x=percent_nodes,y=cdf)
    return g

'''
This method returns the palma coefficient for user events. (User Disparity)

Question #26b

Inputs: DataFrame - Desired dataset


Output: p - The palma coefficient for the dataset
        data - dataframe showing the CDF and Node percentages. (Mainly used for plotting)
'''
def getPalmaCoef(df):
    df = df.copy()
    df.columns = ['time', 'event', 'user', 'repo']
    df['value'] = 1
    df = df.groupby('user')
    event_counts = df.value.sum()


    values = np.sort(np.array(event_counts))


    cdf = np.cumsum(values) / float(np.sum(values))
    percent_nodes = np.arange(len(values)) / float(len(values))


    p10 = np.sum(values[percent_nodes >= 0.9])
    p40 = np.sum(values[percent_nodes <= 0.4])


    p = float(p10) / float(p40)

    x = cdf
    y = percent_nodes
    data = pd.DataFrame({'cum_nodes': y, 'cum_value': x})

    return p

'''
This method returns the top k users with the most events.

Question #24b

Inputs: DataFrame - Desired dataset. Used mainly when dealing with subset of events
        k - Number of users to be returned

Output: Dataframe with the user ids and number of events
'''
def getMostActiveUsers(df,k=10):
    df = df.copy()
    df.columns = ['time', 'event', 'user', 'repo']
    dft = df
    dft['value'] = 1
    dft = df.groupby('user')
    measurement = dft.value.sum().sort_values(ascending=False).head(k)
    measurement = pd.DataFrame(measurement).sort_values('value',ascending=False)
    return measurement

'''
This method returns the distribution for the users activity (event counts).

Question #24a

Inputs: DataFrame - Desired dataset
        eventType - (Optional) Desired event type to use

Output: List containing the event counts per user
'''
def getUserActivityDistribution(df,eventType=None):
    df = df.copy()
    df.columns = ['time', 'event', 'user', 'repo']
    if eventType != None:
        df = df[df.event == eventType]
    df['value'] = 1
    df = df.groupby('user')
    measurement = df.value.sum().reset_index()
    return measurement
