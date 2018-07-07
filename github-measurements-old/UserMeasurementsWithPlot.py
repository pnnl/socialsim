'''
Notice: This computer software was prepared by Battelle Memorial Institute, hereinafter the Contractor, under Contract
No. DE-AC05-76RL01830 with the Department of Energy (DOE).  All rights in the computer software are reserved by DOE on
behalf of the United States Government and the Contractor as provided in the Contract.  You are authorized to use this
computer software for Governmental purposes but it is not to be released or distributed to the public.  NEITHER THE
GOVERNMENT NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS
SOFTWARE.  This notice including this sentence must appear on any copies of this computer software.
'''
from plots import *

'''
The following is the user measurment functions previously released with plotting added. The plots are currently all printed.
'''

'''
This method returns the number of unique repos that a particular set of users contributed too

Question #18

Inputs: DataFrame - Desired dataset
        users - A list of users of interest
        log - to plot with log values default false

Output: A dataframe with the user id and the number of repos contributed to
'''
def getUserUniqueRepos(df,users, log=False):
    df.columns = ['id', 'time', 'event','user', 'repo']
    df = df[df.user.isin(users)]
    df =df.groupby('user')
    data = df.repo.nunique()
    td = data
    print plot_top_users(data,'User','Unique Repos Contributed To','Quantity of Repos Users Contributed To')
    
    return td

'''
This method returns the cumulative activity of the desire user over time.

Question #20

Inputs: DataFrame - Desired dataset
        users - A list of users of interest

Output: A grouped dataframe of the users activity over time
'''
def getUserActivityTimeline(df, users, log=False):
    df.columns = ['id', 'time', 'event','user', 'repo']

    df = df[df.user.isin(users)]
    df['value'] = 1

    df['time'] = pd.to_datetime(df['time'])
    df['time'] = df['time'].dt.strftime('%Y-%m-%d')
    df = df.groupby(['user','time']).sum()

    minDate = df.index.min()[1]
    maxDate = df.index.max()[1]

    idx = pd.date_range(minDate, maxDate)
    ndf = pd.DataFrame()
    first = 0 
    for u in users:
        d = df.loc[u]
        d.index = pd.DatetimeIndex(d.index)
        d = d[['value']].reindex(idx).fillna(0)
        d = d.cumsum()
        d['user'] = u
        d = d.reset_index()
        if first == 0:
            first = 1
            ndf = d
            continue
        ndf = pd.concat([ndf,d])
    ndf.columns = ['time','value','user']
    ndf['time'] = pd.to_datetime(ndf['time'])
    ndf = ndf.sort_values(['time'])
    ndf = ndf.set_index(['time'])

    print plot_activity_timeline(ndf,'Time','Total Number of Contributions','Cumulutive Sum of Contributions')
   
    return ndf


'''
This method returns the top k most popular users for the dataset.

Question #27

Inputs: DataFrame - Desired dataset
        k - (Optional) The number of users that you would like returned.

Output: A dataframe with the user ids and number events for that user
'''
def getUserPopularity(df,k=10, log=False):
    df.columns = ['id', 'time', 'event','user', 'repo']
    df['value'] = 1

    repo_popularity = df[df['event'] != 'CreateEvent'].groupby('repo')['value'].sum().reset_index()
    user_repos = df[df['event'] == 'CreateEvent'].sort_values('time').drop_duplicates(subset='repo',keep='first')
    merged = user_repos[['user','repo']].merge(repo_popularity,on='repo',how='left')
    measurement = merged.groupby('user').value.sum().sort_values(ascending=False).head(k)
    measurement = pd.DataFrame(measurement).sort_values('value',ascending=False)

    print plot_top_users(measurement,'User Popularity','User','User Popularity')

    return measurement


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
This method returns the average time between events for each user

Question #29b and c

Inputs: df - Data frame of all data for repos
        repos - (Optional) List of specific users to calculate the measurement for
        nCPu - (Optional) Number of CPU's to run measurement in parallel

Outputs: A list of average times for each user. Length should match number of repos
'''
def getAvgTimebwEvents(df,users=None, nCPU=1):
    df.columns = ['id','time', 'event', 'user', 'repo']
    df['time'] = pd.to_datetime(df['time'])

    if users == None:
        users = df['user'].unique()

    p = Pool(nCPU)
    mean_time_partial = partial(getMeanTime,df=df)
    deltas = p.map(mean_time_partial,users)
    
    
    _,bins = np.histogram(deltas,bins='auto')
 
    measurement = pd.DataFrame(deltas)

    measurement.plot(kind='hist',bins=bins,legend=False,cumulative=False,normed=False,figsize=(10,7))
    plt.xlabel('Time Between PullRequestEvents in Seconds',fontsize=20)
    plt.ylabel('Number of Repos',fontsize=20)
    plt.title('Average Time Between PullRequestEvents',fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    print plt.show()
    return deltas

'''
This method returns distribution the diffusion delay for each user

Question #29

Inputs: DataFrame - Desired dataset
        unit - (Optional) This is the unit that you want the distribution in. Check np.timedelta64 documentation
        for the possible options

Output: A list (array) of deltas in units specified
'''
def getUserDiffusionDelay(df,unit='s', log=False):
    df.columns = ['id', 'time', 'event', 'user', 'repo']
    df['value'] = df['time']
    df['value'] = pd.to_datetime(df['value'])
    grouped = df.groupby('user')
    transformed = grouped['value'].transform('min')
    delta = df['value'].sub(transformed).apply(lambda x: int(x / np.timedelta64(1, unit)))
    
    print plot_histogram(delta,'User Activity Delay','Number of Users','Diffusion Delay')

    return delta


'''
This method returns the gini coefficient for user events. (User Disparity)

Question #28

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

Question #28

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

    print plot_palma(data,'Cumulative share of Repos','Cumulative share of Events','User Event Dispartiy')
    
    return p,data

'''
This method returns the top k users with the most events.

Question #26b

Inputs: DataFrame - Desired dataset. Used mainly when dealing with subset of events
        k - Number of users to be returned

Output: Dataframe with the user ids and number of events
'''
def getMostActiveUsers(df,k=10, log=True):
    df.columns = ['id', 'time', 'event', 'user', 'repo']
    df['value'] = 1
    df = df.groupby('user')
    measurement = df.value.sum().sort_values(ascending=False).head(k)
    measurement = pd.DataFrame(measurement).sort_values('value',ascending=False)

    print plot_top_users(measurement,'User','User Activity','Top Users')


'''
This method returns the distribution for the users activity (event counts).

Question #26a

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

    print plot_histogram(d.value.values,'Total Activity','Number of Users','User Activity Distribution')

    return np.array(measurement).tolist()