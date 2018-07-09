import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import Pool
from functools import partial
from pathos import pools as pp

'''
This class implements user centric method. Each function will describe which metric it is used for according
to the questions number and mapping.
These metrics assume that the data is in the order id,created_at,type,actor.id,repo.id
'''

class UserCentricMeasurements(object):
    def __init__(self):
        super(UserCentricMeasurements, self).__init__()

    '''
    This function selects a subset of the full data set for a selected set of users and event types.
    Inputs: users - A boolean or a list of users.  If it is list of user ids (login_h) the data frame is subset on only this list of users.
                    If it is True, then the pre-selected node-level subset is used.  If False, then all users are included.
            eventType - A list of event types to include in the data set

    Output: A data frame with only the selected users and event types.
    '''
    def determineDf(self,users,eventType):

        if users == True:
            #self.selectedUsers is a data frame containing only the users in interested_users
            df = self.selectedUsers
        elif users != False:
            df = df[df.user.isin(users)]
        else:
            df = self.main_df

        if eventType != None:
            df = df[df.event.isin(eventType)]

        return df

    '''
    This method returns the number of unique repos that a particular set of users contributed too
    Question #17
    Inputs: selectedUsers - A list of users of interest or a boolean indicating whether to subset to the node-level measurement users.
            eventType - A list of event types to include in the data
    Output: A dataframe with the user id and the number of repos contributed to
    '''
    def getUserUniqueRepos(self,selectedUsers=False,eventType=None):
        df = self.determineDf(selectedUsers,eventType)
        df = df.groupby('user')
        data = df.repo.nunique().reset_index()
        data.columns = ['user','value']
        return data

    '''
    This method returns the timeline of activity of the desired user over time, either in raw or cumulative counts.
    Question #19
    Inputs: selectedUsers - A list of users of interest or a boolean indicating whether to subset to node-level measurement users.
            time_bin - Time frequency for calculating event counts
            cumSum - Boolean indicating whether to calculate the cumulative activity counts
            eventType = List of event types to include in the data
    Output: A dictionary with a data frame for each user with two columns: data and event counts
    '''
    def getUserActivityTimeline(self, selectedUsers=True,time_bin='1d',cumSum=False,eventType=None):
        df = self.determineDf(selectedUsers,eventType)

        df['value'] = 1
        if cumSum:
            df['cumsum'] = df.groupby('user').value.transform(pd.Series.cumsum)
            df = df.groupby(['user',pd.Grouper(key='time',freq=time_bin)]).max().reset_index()
            df['value'] = df['cumsum']
            df = df.drop('cumsum',axis=1)
        else:
            df = df.groupby(['user',pd.Grouper(key='time',freq=time_bin)]).sum().reset_index()

        data = df.sort_values(['user', 'time'])
        measurements = {}
        for user in data['user'].unique():
            measurements[user] = data[data['user'] == user]

        return measurements


    '''
    This method returns the top k most popular users for the dataset, where popularity is measured
    as the total popularity of the repos created by the user.
    Question #25
    Inputs: k - (Optional) The number of users that you would like returned.
            use_metadata - External metadata file containing repo owners.  Otherwise use first observed user with 
                           a creation event as a proxy for the repo owner.
            eventType - A list of event types to include
    Output: A dataframe with the user ids and number events for that user
    '''
    def getUserPopularity(self,k=5000,use_metadata=False,eventType=None):

        df = self.determineDf(False,eventType)

        df['value'] = 1

        repo_popularity = df.groupby('repo')['value'].sum().reset_index()

        if use_metadata:
            #merge repo popularity with the owner information in repo_metadata
            #drop data for which no owner information exists in metadata
            merged = repo_popularity.merge(self.repoMetaData,left_on='repo',right_on='full_name_h',
                                           how='left').dropna()
        elif df['repo'].str.match('.{22}/.{22}').all():
            #if all repo IDs have the correct format use the owner info from the repo id
            repo_popularity['owner_id'] = repo_popularity['repo'].apply(lambda x: x.split('/')[0])
        else:
            #otherwise use creation event as a proxy for ownership
            user_repos = df[df['event'] == 'CreateEvent'].sort_values('time').drop_duplicates(subset='repo',keep='first')
            user_repos = user_repos[['user','repo']]
            user_repos.columns = ['owner_id','repo']
            if len(user_repos.index) >= 0:
                repo_popularity = user_repos.merge(repo_popularity,on='repo',how='left')
            else:
                return None

        measurement = repo_popularity.groupby('owner_id').value.sum().sort_values(ascending=False).head(k)
        measurement = pd.DataFrame(measurement).sort_values('value',ascending=False)
        return measurement


    '''
    This method returns the average time between events for each user

    Inputs: df - Data frame of all data for repos
    users - (Optional) List of specific users to calculate the metric for
    nCPu - (Optional) Number of CPU's to run metric in parallel
    Outputs: A list of average times for each user. Length should match number of repos
    '''
    def getAvgTimebwEventsUsers(self,selectedUsers=True, nCPU=1):
        df = self.determineDf(selectedUsers)
        users = self.df['user'].unique()
        args = [(df, users[i]) for i, item_a in enumerate(users)]
        pool = pp.ProcessPool(nCPU)
        deltas = pool.map(self.getMeanTimeHelper, args)
        return deltas

    '''
    Helper function for getting the average time between events

    Inputs: Same as average time between events
    Output: Same as average time between events
    '''
    def getMeanTimeUser(self,df, user):
        d = df[df.user == user]
        d = d.sort_values(by='time')
        delta = np.mean(np.diff(d.time)) / np.timedelta64(1, 's')
        return delta

    def getMeanTimeUserHelper(self,args):
        return self.getMeanTimeUser(*args)

    '''
    This method returns distribution the diffusion delay for each user
    Question #27
    Inputs: DataFrame - Desired dataset
    unit - (Optional) This is the unit that you want the distribution in. Check np.timedelta64 documentation
    for the possible options
    metadata_file - File containing user account creation times.  Otherwise use first observed action of user as proxy for account creation time.
    Output: A list (array) of deltas in units specified
    '''
    def getUserDiffusionDelay(self,unit='h', selectedUser=True,eventType=None):

        df = self.determineDf(selectedUser,eventType)

        df['value'] = df['time']
        df['value'] = pd.to_datetime(df['value'])
        df['value'] = df['value'].dt.round('1H')

        if self.useUserMetaData:
            df = df.merge(self.userMetaData[['user','created_at']],left_on='user',right_on='user',how='left')
            df = df[['user','created_at','value']].dropna()
            measurement = df['value'].sub(df['created_at']).apply(lambda x: int(x / np.timedelta64(1, unit)))
        else:
            grouped = df.groupby('user')
            transformed = grouped['value'].transform('min')
            measurement = df['value'].sub(transformed).apply(lambda x: int(x / np.timedelta64(1, unit)))
        return measurement

    '''
    This method returns the top k users with the most events.
    Question #24b
    Inputs: DataFrame - Desired dataset. Used mainly when dealing with subset of events
    k - Number of users to be returned
    Output: Dataframe with the user ids and number of events
    '''
    def getMostActiveUsers(self,k=5000,eventType=None):

        df = self.main_df

        if eventType != None:
            df = df[df.event.isin(eventType)]

        df['value'] = 1
        df = df.groupby('user')
        measurement = df.value.sum().sort_values(ascending=False).head(k)
        measurement = pd.DataFrame(measurement).sort_values('value',ascending=False)
        return measurement

    '''
    This method returns the distribution for the users activity (event counts).
    Question #24a
    Inputs: DataFrame - Desired dataset
    eventType - (Optional) Desired event type to use
    Output: List containing the event counts per user
    '''
    def getUserActivityDistribution(self,eventType=None,selectedUser=False):

        if selectedUser:
            df = self.selectedUsers
        else:
            df = self.main_df

        if eventType != None:
            df = df[df.event.isin(eventType)]

        df['value'] = 1
        df = df.groupby('user')
        measurement = df.value.sum().reset_index()
        return measurement


    '''
    Calculate the proportion of pull requests that are accepted by each user.
    Question #15 (Optional Measurement)
    Inputs: eventType: List of event types to include in the calculation (Should be PullRequestEvent).
            thresh: Minimum number of PullRequests a repo must have to be included in the distribution. 
    Output: Data frame with the proportion of accepted pull requests for each user
    '''
    def getUserPullRequestAcceptance(self,eventType=['PullRequestEvent'], thresh=2):

        df = self.main_df_opt

        if not df is None and 'PullRequestEvent' in self.main_df.event.values:
            df = df[self.main_df.event.isin(eventType)]
            users_repos = self.main_df[self.main_df.event.isin(eventType)]

            #subset on only PullRequest close actions (not opens)
            idx = df['action'] == 'closed'
            closes = df[idx]
            users_repos = users_repos[idx]

            #merge pull request columns (action, merged) with main data frame columns
            closes = pd.concat([users_repos,closes],axis=1)
            closes = closes[['user','repo','merged']]
            closes['value'] = 1

            #add up number of accepted (merged) and rejected pullrequests by user and repo
            outcomes = closes.pivot_table(index=['user','repo'],values=['value'],columns=['merged'],aggfunc=np.sum).fillna(0)
            
            outcomes.columns = outcomes.columns.get_level_values(1)

            outcomes = outcomes.rename(index=str, columns={True: "accepted", False: "rejected"})

            for col in ['accepted','rejected']:
                if col not in outcomes.columns:
                    outcomes[col] = 0


            outcomes['total'] = outcomes['accepted'] +  outcomes['rejected']
            outcomes['value'] = outcomes['accepted'] / outcomes['total']
            outcomes = outcomes.reset_index()
            outcomes = outcomes[outcomes['total'] >= thresh]


            if len(outcomes.index) > 0:
                #calculate the average acceptance rate for each user across their repos
                measurement = outcomes[['user','value']].groupby('user').mean()
            else:
                measurement = None
        else:
            measurement = None


        return measurement
    
