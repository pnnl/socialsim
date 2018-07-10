from __future__ import division
import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import Pool
from functools import partial
from pathos import pools as pp
from multiprocessing import Pool

'''
This class implements repo centric methods.
These metrics assume that the data is in the order id,created_at,type,actor.id,repo.id
'''

class RepoCentricMeasurements(object):
    def __init__(self):
        super(RepoCentricMeasurements, self).__init__()


    '''
    This function creates a dictionary of data frames with
    each entry being the activity of one repo from the repos
    argument.

    This is used for the selected repos for the node-level meausurements. 
    Inputs: repos - List of repo ids (full_name_h)
    Output: Dictionary of data frames with the repo ids as the keys
    '''
    def getSelectRepos(self, repos):
        reposDic = {}
        for ele in repos:
            d = self.main_df[self.main_df['repo'] == ele]
            reposDic[ele] = d
        return reposDic

    '''
    This function runs a particular measurement (method) on the
    repos that were selected by getSelectRepos.

    This is used for the selected repos for the node-level meausurements. 

    Inputs: method - Measurement function
    Output: Dictionary of measurement results with the repo ids as the keys
    '''
    def runSelectRepos(self, method, *args):

        ans = {}
        for ele in self.selectedRepos.keys():
            df = self.selectedRepos[ele].copy()
            ans[ele] = method(df,*args)
        return ans


    '''
    A wrapper function to calculate the distributon for the diffusion delay for each node.
    Question #1
    Inputs: DataFrame - Data
        eventType - A list of events to filter data on
        unit - Time unit for time differences, e.g. "s","d","h"
        selectedRepos - If True calculate the measurement on the each of the selected repo nodes
                        otherwise calculate it on the full data set
    Output: An array of deltas in the given units
    '''    
    def getRepoDiffusionDelay(self, eventType=None, unit='h', selectedRepos=True):
        if selectedRepos:
            return self.runSelectRepos(self.getRepoDiffusionDelayHelper, eventType, unit)
        else:
            return self.getRepoDiffusionDelayHelper(self.main_df, eventType,unit)


    '''
    This method returns the distributon for the diffusion delay.
    Question #1
    Inputs: DataFrame - Data
        eventType - A list of events to filter data on
        unit - Time unit for time differences, e.g. "s","d","h"
    Output: An array of deltas in the given units
    '''
    def getRepoDiffusionDelayHelper(self, df, eventType=None, unit='h'):

        if eventType != None:
            df = df[df.event.isin(eventType)]
            

        if len(df.index) == 0: 
            return None

        repo = df['repo'].iloc[0]

        #use metadata for repo creation dates if available
        if self.useRepoMetaData:
            df = df.merge(self.repoMetaData,left_on='repo',right_on='repo',how='left')
            df = df[['repo','created_at','time']].dropna()
            df['delta'] = (df['time']-df['created_at']).apply(lambda x: int(x / np.timedelta64(1, unit)))
        #otherwise use first observed activity as a proxy
        else:
            creation_day = df['time'].min()
            df['delta'] = (df['time']-creation_day).apply(lambda x: int(x / np.timedelta64(1, unit)))
            df = df.iloc[1:]

        delta = df['delta'].values
        return delta


    '''
    A wrapper function the growth of a repo over time for each repo.
    Question #2
    Input: df - Dataframe of all data for a repo
       cumSum - This is a boolean that indicates if the dataframe should be cumuluative over time
       selectedRepos - If True calculate the measurement on the each of the selected repo nodes
                        otherwise calculate it on the full data set 
    output - A dataframe that describes the repo growth. Indexed on time.
    '''
    def getRepoGrowth(self, eventType=None, cumSum=False, selectedRepos=True):

        if selectedRepos:
            return self.runSelectRepos(self.getRepoGrowthHelper, eventType, cumSum)
        else:
            return self.getRepoGrowthHelper(self.main_df, eventType, cumSum)

    '''
    This method returns the growth of a repo over time.
    Question #2
    Input: df - Dataframe of all data for a repo
           cumSum - This is a boolean that indicates if the dataframe should be cumuluative over time.
    output - A dataframe that describes the repo growth. Indexed on time.
    '''
    def getRepoGrowthHelper(self, df, eventType=None, cumSum=False):

        if eventType != None:
             df = df[df.event.isin(eventType)]

        df = df.set_index("time")

        p = df[['repo']].groupby(pd.TimeGrouper('D')).count()

        p.columns = ['value']

        if cumSum == True:
            p['value'] = p.cumsum(axis=0)['value']

        return p.reset_index()


    '''
    Wrapper function to calculate the total number of unique daily contributers to a repo or 
    the unique daily contributors who are new contributors for each selected repo
    Question # 4
        Input: df - Events data frame
               newUsersOnly - Boolean to indicate whether to calculate total daily unique users (False) or daily new contributers (True), 
                              if None run both total and new unique users.
               cumulative - Boolean to indicate whether or not the metric should be cumulative over time
               eventType - A list of event types to include in the calculation
        Output: A dictionary containing data frame with daily event counts for each repo
    '''
    def getContributions(self,eventType=None,newUsersOnly=False,cumulative=False,selectedRepos=True):
        if selectedRepos:
            return self.runSelectRepos(self.getContributionsHelper, newUsersOnly, cumulative,eventType)
        else:
            return self.getContributionsHelper(self.main_df,newUsersOnly,cumulative, eventType)


    '''
    Calculates the total number of unique daily contributers to a repo or the unique daily contributors who are new contributors
    Question # 4
        Input: df - Events data frame
               newUsersOnly - Boolean to indicate whether to calculate total daily unique users (False) or daily new contributers (True), 
                              if None run both total and new unique users.
               cumulative - Boolean to indicate whether or not the metric should be cumulative over time
               eventType - A list of event types to include in the calculation
        Output: A data frame with daily event counts
    '''
    def getContributionsHelper(self,df, newUsersOnly, cumulative, eventType):

        def contributionsInsideHelper(dfH,newUsersOnly,cumulative):
            if newUsersOnly:
                #drop duplicates on user so a new user only shows up once in the data
                dfH = dfH.drop_duplicates(subset=['user'])
 
            p = dfH[['user']].groupby(pd.TimeGrouper('D')).nunique()
            
            if cumulative:
                #get cumulative user counts
                p['user'] = p.cumsum(axis=0)['user']
            
            p = p.reset_index()
            p.columns = ['user','value']
            return p

        if eventType != None:
            df = df[df.event.isin(eventType)]

        df = df.set_index("time")

        if newUsersOnly == None:
            #run both total daily user counts and daily new user counts
            new_users = contributionsInsideHelper(df, True, cumulative)
            total_users = contributionsInsideHelper(df,False, cumulative)

            new_users = new_users.reset_index(drop=True)
            total_users = total_users.reset_index(drop=True)

            return new_users, total_users

        else:
            #run only one of total daily counts or new daily counts
            results = contributionsInsideHelper(df,newUsersOnly, cumulative)
            return results

    
    def getDistributionOfEvents(self,nCPU = 4,weekday=False, selectedRepos=True):
        if selectedRepos == True:
            return self.runSelectRepos(self.getDistributionOfEventsHelper, nCPU, weekday)
        else:
            return self.getDistributionOfEventsHelper(self.main_df,nCPU,weekday)

    '''
    This method returns the distribution for each event over time or by weekday. Default is over time.
    Question #5
    Inputs: df - Data frame of all data for repos
            nCPu - (Optional) Number of CPU's to run metric in parallel
            weekday - (Optional) Boolean to indicate whether the distribution should be done by weekday. Default is False.
    Output: Dataframe with the distribution of events by weekday. Columns: Event, Weekday, Count or Event, Date, Count
    '''
    def getDistributionOfEventsHelper(self,df,nCPU, weekday=False):
        df['id'] = df.index
        df['weekday'] = df['time'].dt.weekday_name
        df['date'] = df['time'].dt.date

        if weekday:
            col = 'weekday'
        else:
            col = 'date'

        counts = df.groupby(['event',col])['user'].count().reset_index()
        counts.columns = ['event',col,'value']

        return counts

    '''
    Helper Function for getting the Dist. of Events per weekday.
    '''
    def processDistOfEvents(self,df,weekday):

        df.set_index('time', inplace=True)
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['year'] = df.index.year

        if weekday:
            df['weekday'] = df.apply(lambda x:datetime(x['year'],x['month'],x['day']).weekday(),axis=1)
            p = df[['event','user','weekday']].groupby(['event','weekday']).count()
            p = p.reset_index()
            return p

        else:
            p = df[['event', 'year', 'month', 'day','id']].groupby(['event', 'year', 'month','day']).count()
            p = pd.DataFrame(p).reset_index()
            p.column = ['event', 'year', 'month','day','count']
            p['date'] = p.apply(lambda x: datetime.strptime("{0} {1} {2}".format(x['year'], x['month'],x['day']), "%Y %m %d"), axis=1)
            p['date'] = p['date'].dt.strftime('%Y-%m-%d')
            p = p.reset_index()
            return p


    '''
    Wrapper function calculate the gini coefficient for the data frame.
    Question #6,14,26
    Input: df - Data frame containing data can be any subset of data
           nodeType - Type of node to calculate the Gini coefficient over.  Options: user or repo (case sensitive)
           eventType - A list of event types to include in the calculation
    Output: g - gini coefficient
    '''
    def getGiniCoef(self,nodeType='repo', eventType=None):
            return self.getGiniCoefHelper(self.main_df, nodeType,eventType)


    '''
    This method returns the gini coefficient for the data frame.
    Question #6,14,26
    Input: df - Data frame containing data can be any subset of data
           nodeType - Type of node to calculate the Gini coefficient over.  Options: user or repo (case sensitive)
           eventType - A list of event types to include in the calculation
    Output: g - gini coefficient
    '''
    def getGiniCoefHelper(self, df, nodeType,eventType):

        if eventType is not None:
            df = df[df.event.isin(eventType)]

        #count events for given node type
        df = df[['repo', 'user']].groupby(nodeType).count()
        df.columns = ['value']
        df = df.reset_index()

        values = df['value'].values
        values = np.sort(np.array(values))
        
        #cumulative event counts
        cdf = np.cumsum(values) / float(np.sum(values))
        #cumulative node counts
        percent_nodes = np.arange(len(values)) / float(len(values))
                
        #area between Lorenz curve and line of equality
        g = 1 - 2*np.trapz(x=percent_nodes,y=cdf)
        
        return g

    '''
    A wrapper function to calculate the Palma coefficient.
    Question #6,14,26
    Input: nodeType - (Optional) This is the node type on whose event counts the Palma coefficient 
                      is calculated. Options: user or repo (case sensitive)
           eventType - A list of event types to include in the calculation
    Output: p - Palma Coefficient
    '''    
    def getPalmaCoef(self,nodeType='repo', eventType=None):
        return self.getPalmaCoefHelper(self.main_df, nodeType, eventType)

    '''
    This method returns the Palma coefficient.
    Question #6,14,26
    Input: df - Data frame containing data can be any subset of data
           nodeType - (Optional) This is the node type on whose event counts the Palma coefficient 
                      is calculated. Options: user or repo (case sensitive)
           eventType - A list of event types to include in the calculation
    Output: p - Palma Coefficient
    '''
    def getPalmaCoefHelper(self, df, nodeType='repo', eventType=None):

        if eventType is not None:
            df = df[df.event.isin(eventType)]

        df = df[['repo', 'user']].groupby(nodeType).count()

        df.columns = ['value']
        df = df.reset_index()

        values = df['value'].values
        values = np.sort(np.array(values))
        percent_nodes = np.arange(1, len(values) + 1) / float(len(values))


        #percent of events taken by top 10% of nodes
        p10 = np.sum(values[percent_nodes >= 0.9])
        #percent of events taken by bottom 40% of nodes
        p40 = np.sum(values[percent_nodes <= 0.4])

        try:
            p = float(p10) / float(p40)
        except ZeroDivisionError:
            return None

        return p


    '''
    This method returns the top-k repos by event count for selected event types
    Question #12,13
    Inputs: df - Data frame with events data
            eventType - A list of event types to include in the calculation
    Outputs: Dataframe with the top-k repos and their event counts. Columns are repo id and the count of that event.
    '''
    def getTopKRepos(self,k=100,eventType=['WatchEvent']):
        df = self.main_df
        df = df[df.event.isin(eventType)]
        p = df[['repo', 'event']].groupby(['repo']).count()
        p = p.sort_values(by='event',ascending=False)
        p.columns = ['value']
        return p.head(k)


    '''
    This method returns the distribution of event type per repo e.g. x repos with y number of events, z repos with n
    amounts of events.
    Question #11,12,13
    Inputs: df - Data frame with data for all repos
            eventType - List of event type(s) to get distribution over
    Outputs: Dataframe with the distribution of event type per repo. Columns are repo id and the count of that event.
    '''
    def getDistributionOfEventsByRepo(self,eventType=['WatchEvent']):

        df = self.main_df
        if eventType != None:
            df = df[df['event'].isin(eventType)]

        p = df[['repo','time']].groupby('repo').count()
        p = p.sort_values(by='time')
        p.columns = ['value']
        p = p.reset_index()
        return p

    '''
    A wrapper function to calculate the average time between events for each repo
    NOTE: Multithreading is highly recommended for datasets with more than 5000 repos.
    Question #11
    Inputs: repos - (Optional) Boolean to use selected repo nodes.  If False, calculate for all repos.
            eventType - List of event type(s) to get distribution over
    Outputs: A list of average times for each repo. Length should match number of repos. Elements with NaN correspond to a
             repo only having a single event.
    '''
    def getAvgTimebtwEvents(self, eventType=None, repos=False):
        df = self.main_df

        if eventType != None:
            df = df[df.event.isin(eventType)]

        if repos:
            repo_list = self.selectedRepos.keys()
            df = df[df.repo.isin(repo_list)]

        deltas = df.groupby('repo')['time'].apply(self.getMeanTimeHelper)

        return deltas

    '''
    Calculates the average time between events for each repo
    Question #12
    Inputs:  times - A Series containing the event times for an individual repo
    Outputs: Average time between events for the repo in hours
    '''
    def getMeanTimeHelper(self, times):
        delta = np.mean(times.diff()) / np.timedelta64(1, 'h')
        return delta


    '''
    Calculate the proportion of pull requests that are accepted for each repo.
    Question #15 (Optional Measurement)
    Inputs: eventType: List of event types to include in the calculation (Should be PullRequestEvent).
            thresh: Minimum number of PullRequests a repo must have to be included in the distribution.
    Output: Data frame with the proportion of accepted pull requests for each repo 
    '''
    def getRepoPullRequestAcceptance(self,eventType=['PullRequestEvent'],thresh=2):

        df = self.main_df_opt

        #check if optional columns exist
        if not df is None and 'PullRequestEvent' in self.main_df.event.values:
            df = df[self.main_df.event.isin(eventType)]
            users_repos = self.main_df[self.main_df.event.isin(eventType)]

            #subset to only pull requests which are being closed (not opened)
            idx = df['action'] == 'closed'
            closes = df[idx]
            users_repos = users_repos[idx]

            #merge optional columns (action, merged) with the main data frame columns
            closes = pd.concat([users_repos,closes],axis=1)
            closes = closes[['repo','merged']]
            closes['value'] = 1

            #create count of accepted (merged) and rejected pull requests by repo
            outcomes = closes.pivot_table(index=['repo'],values=['value'],columns=['merged'],aggfunc='sum').fillna(0)
            
            outcomes.columns = outcomes.columns.get_level_values(1)
            
            outcomes = outcomes.rename(index=str, columns={True: "accepted", False: "rejected"})

            #if only accepted or reject observed in data, create other column and fill with zero
            for col in ['accepted','rejected']:
                if col not in outcomes.columns:
                    outcomes[col] = 0

            #get total number of pull requests per repo by summing accepted and rejected
            outcomes['total'] = outcomes['accepted'] + outcomes['rejected']
            #get proportion
            outcomes['value'] = outcomes['accepted'] / outcomes['total']

            #subset on repos which have enough data
            outcomes = outcomes[outcomes['total'] >= thresh]

            if len(outcomes.index) > 0:
                measurement = outcomes
            else:
                measurement = None
        else:
            measurement = None


        return measurement

    def getIssueVsPushProbability(self,selectedRepos=True,eventType=None):
        if selectedRepos == True:
            return self.runSelectRepos(self.getIssueVsPushProbabilityHelper,eventType)
        else:
            return self.getIssueVsPushProbability(self.main_df,eventType)

    def getIssueVsPushProbabilityHelper(self,df,eventType):

        if eventType != None:
            df = df[df['event'].isin(eventType)]

        df['value'] = 1

        def user_repo_cumulative_count(grp):

            grp['value'] = grp.value.cumsum()

            return(grp)

        if len(df.index) < 1:
            return None

        measurement = df.groupby(['repo','user']).apply(user_repo_cumulative_count).reset_index()

        if self.previous_event_counts != None:
            measurement = measurement.merge(self.previous_event_counts,on=['user','repo'],how='left').fillna(0)
            measurement['value'] = measurement['value'] + measurement['count']

        measurement = measurement[measurement['event'].isin(['IssuesEvent','PushEvent'])]

        measurement['issue'] = measurement['event'] == 'IssuesEvent'
        measurement['push'] = measurement['event'] == 'PushEvent'


        measurement['next_event_issue'] = measurement['issue'].shift(-1)
        measurement['next_event_push'] = measurement['push'].shift(-1)


        bins = np.logspace(-1,3.0,16)
        measurement['num_events_binned'] = pd.cut(measurement['value'],bins).apply(lambda x: np.floor(x.right)).astype(float)


        def ratio(grp):

            if float(grp['next_event_issue'].sum()) > 0:
                return float(grp['next_event_push'].sum()) / float(grp['next_event_issue'].sum())
            else:
                return 0.0

        if len(measurement.index) > 0:
            measurement = measurement.groupby(['repo','num_events_binned']).apply(ratio).reset_index()
            measurement.columns = ['repo','num_events_binned','value']
        else:
            measurement = None

        return(measurement)

    def propUserContinue(self, eventType=None, selectedRepos=True):

        if selectedRepos:
            return self.runSelectRepos(self.propUserContinueHelper, eventType)
        else:
            return self.propUserContinueHelper(self.main_df, eventType)


    def propUserContinueHelper(self,df,eventType):
        
        if not eventType is None:            
            data = df[df['event'].isin(eventType)]


        if len(data.index) > 1:
            data['value'] = 1
            grouped = data.groupby(['user','repo'])
            if grouped.ngroups > 1:
                measurement = grouped.apply(lambda grp: grp.value.cumsum()).reset_index()
            else:
                data['value'] = data['value'].cumsum()
                measurement = data.copy()
            grouped = measurement.groupby(['user','repo']).value.max().reset_index()
            grouped.columns = ['user','repo','num_events']
            measurement = measurement.merge(grouped,on=['user','repo'])
            measurement['last_event'] = measurement['value'] == measurement['num_events']
            
            bins = np.logspace(-1,2.5,30)
            measurement['num_actions'] = pd.cut(measurement['value'],bins).apply(lambda x: np.floor(x.right)).astype(float)
            measurement['last_event'] = ~measurement['last_event']
            measurement = measurement.groupby(['repo','num_actions']).last_event.mean().reset_index()
            measurement.columns = ['repo','num_actions','value']
        else:
            measurement = None

        return measurement
