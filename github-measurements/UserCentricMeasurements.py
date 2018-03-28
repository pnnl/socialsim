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
This class implements user centric method. Each function will describe which measurement it is used for according
to the questions number and mapping.

These measurements assume that the data is in the order id,created_at,type,actor.id,repo.id
'''


'''
This method returns the number of unique repos that a particular set of users contributed too

Question #17

Inputs: DataFrame - Desired dataset
        users - A list of users of interest

Output: A dataframe with the user id and the number of repos contributed to
'''
def getUserUniqueRepos(df,users):
    df.columns = ['id', 'time', 'event','user', 'repo']
    df = df[df.user.isin(users)]
    df =df.groupby('user')
    data = df.repo.nunique()
    return data


'''
This method returns the cumulative activity of the desire user over time.

Question #19

Inputs: DataFrame - Desired dataset
        users - A list of users of interest

Output: A grouped dataframe of the users activity over time
'''
def getUserActivityTimeline(df, users):
    df.columns = ['id', 'time', 'event','user', 'repo']

    df = df[df.user.isin(users)]
    df['value'] = 1
    df['cumsum'] = df.groupby('user').value.transform(pd.Series.cumsum)
    data = df[['user', 'time', 'cumsum']].sort_values(['user', 'time'])
    return data

'''
This method returns the top k most popular users for the dataset.

Question #25

Inputs: DataFrame - Desired dataset
        k - (Optional) The number of users that you would like returned.

Output: A dataframe with the user ids and number events for that user
'''
def getUserPopularity(df,k=10):
    df.columns = ['id', 'time', 'event','user', 'repo']
    df['value'] = 1
    repo_popularity = df[df['event'] != 'CreateEvent'].groupby('repo')['value'].sum().reset_index()
    user_repos = df[df['event'] == 'CreateEvent'].sort_values('time').drop_duplicates(subset='repo',keep='first')
    merged = user_repos[['user','repo']].merge(repo_popularity,on='repo',how='left')
    measurement = merged.groupby('user').value.sum().sort_values(ascending=False).head(k)
    measurement = pd.DataFrame(measurement).sort_values('value',ascending=False)
    return measurement


'''
This method returns distribution the diffusion delay for each user

Question #29

Inputs: DataFrame - Desired dataset
        unit - (Optional) This is the unit that you want the distribution in. Check np.timedelta64 documentation
        for the possible options

Output: A list (array) of deltas in units specified
'''
def getUserDiffusionDelay(df,unit='s'):
    df.columns = ['id', 'time', 'event', 'user', 'repo']
    df['value'] = df['time']
    df['value'] = pd.to_datetime(df['value'])
    grouped = df.groupby('user')
    transformed = grouped['value'].transform('min')
    measurement = df['value'].sub(transformed).apply(lambda x: int(x / np.timedelta64(1, unit)))
    return measurement


'''
This method returns the gini coefficient for user events. (User Disparity)

Question #26

Inputs: DataFrame - Desired dataset


Output: The gini coefficient for the dataset
'''
def getGiniCoef(df):
    df.columns = ['id', 'time', 'event', 'user', 'repo']
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

Question #26

Inputs: DataFrame - Desired dataset


Output: p - The palma coefficient for the dataset
        data - dataframe showing the CDF and Node percentages. (Mainly used for plotting)
'''
def getPalmaCoef(df):
    df.columns = ['id', 'time', 'event', 'user', 'repo']
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

    return p,data

'''
This method returns the top k users with the most events.

Question #24

Inputs: DataFrame - Desired dataset. Used mainly when dealing with subset of events
        k - Number of users to be returned

Output: Dataframe with the user ids and number of events
'''
def getMostActiveUsers(df,k=10):
    df.columns = ['id', 'time', 'event', 'user', 'repo']
    df['value'] = 1
    df = df.groupby('user')
    measurement = df.value.sum().sort_values(ascending=False).head(k)
    measurement = pd.DataFrame(measurement).sort_values('value',ascending=False)
    return measurement

'''
This method returns the distribution for the users activity (event counts).

Question #24

Inputs: DataFrame - Desired dataset
        eventType - (Optional) Desired event type to use

Output: List containing the event counts per user
'''
def getUserActivityDistribution(df,eventType=None):
    df.columns = ['id', 'time', 'event', 'user', 'repo']
    if eventType != None:
        df = df[df.event == eventType]
    df['value'] = 1
    df = df.groupby('user')
    measurement = df.value.sum()
    return np.array(measurement).tolist()