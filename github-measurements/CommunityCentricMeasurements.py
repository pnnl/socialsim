import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import Pool
from functools import partial
from pathos import pools as pp
import pickle as pkl
import warnings

'''
This class implements community centric method. Each function will describe which metric it is used for according
to the questions number and mapping.
These metrics assume that the data is in the order id,created_at,type,actor.id,repo.id
'''

class CommunityCentricMeasurements():
    def __init__(self):
        super(CommunityCentricMeasurements, self).__init__()

    '''
    This method splits the user meta data into location and creation date data frames
    '''
    def loadMetaData(self):
        if self.useUserMetaData:
            self.created_at_df = self.userMetaData[['user','created_at']]
            self.locations_df = self.userMetaData[['user','location']]
            
    '''
    This method loads the community dictionary from the specified pickle file.
    The pickle file should contain a dictionary of the format
    {"type_of_community1":{"community1":[repo1,repo2,...], "community2":[repo3,repo4,...]},
     "type_of_community2":...}
     Inputs: path - file path of pickle file
    '''
    def loadCommunities(self,path):
        with open(path, 'rb') as handle:
            self.comDic = pkl.load(handle)

    '''
    This method subsets the full data frame based on community membership of repos or users.
    Inputs: path - file path to pickle file containing community lists
    Outputs: A dictionary containing a data frame for each community
    '''
    def getCommunities(self,path='data/communities.pkl'):
        self.loadMetaData() 
        self.loadCommunities(path)
        comValuesDic = {}
        repoOrent = ['languages','topics']
        userOrent = ['location','companies']
        
        #repo-focused communities
        for community in repoOrent:
            if community in self.comDic.keys():
                 for key in self.comDic[community]:
                    d = self.main_df[self.main_df['repo'].isin(self.comDic[community][key])]
                    comValuesDic[key] = d

        #user-focused communities
        for community in userOrent:
            if community in self.comDic.keys():
                 for key in self.comDic[community]:
                    d = self.main_df[self.main_df['user'].isin(self.comDic[community][key])]
                    comValuesDic[key] = d
                    
        return comValuesDic

    '''
    A function to calculate a specified measurement on each community.
    Inputs: method - The measurement function
    Output: A dictionary containing the measurement result for each community
    '''
    def runCommunities(self, method, *args):
        ans = {}
        for ele in self.communities.keys():
            df = self.communities[ele]
            ans[ele] = method(df,*args)
        return ans

    '''
    A wrapper function to calculate the proportion of each event type in each community.
    Question #7
    Inputs: communities - Boolean to calculate the measurement seperately for each community (True) or for the full data set (False)
            eventType - List of event types to include
    Output: Dictionary of data frames with columns for event type and proportion, with one data frame for each community
    '''
    def getProportion(self, communities=True,eventType=None):
        if communities:
            return self.runCommunities(self.getProportionHelper,eventType)
        else:
            return self.getProportionHelper(self.main_df)
   
    '''
    Calculates the proportion of each event type in the data.
    Question #7
    Inputs: df - Events data frame
            communities - Boolean to calculate the measurement seperately for each community (True) or for the full data set (False)
            eventType - List of event types to include
    Output: Dictionary of data frames with columns for event type and proportion, with one data frame for each community
    '''
    def getProportionHelper(self,df,eventType):

        if eventType != None:
            df = df[df['event'].isin(eventType)]

        p = df[['user','event']].groupby('event').count()
        p = p.reset_index()
        p.columns = ['event', 'value']

        p['value'] = p['value']/p.value.values.sum()

        return p

    '''
    A wrapper function to calculate the proportion of users who interact with a community who are active contributors.
    Question #20
    Inputs: communities - Boolean to calculate the measurement seperately for each community (True) or for the full data set (False)
    Output: Dictionary of values for each community
    '''
    def contributingUsers(self, communities=True):
        if communities:
            return self.runCommunities(self.contributingUsersHelper)
        else:
            return self.contributingUsersHelper(self.main_df)

    '''
    This method calculates the proportion of users with events in teh data who are active contributors.
    Question #20
    Inputs: df - Events data
    Output: Proportion of all users who make active contributions
    '''
    def contributingUsersHelper(self, df):

        #total number of unique users
        totalUsers = df['user'].nunique()
        df = df[df['event'].isin(self.contribution_events)]

        #number of unique users with direct contributions
        contribUsers = df['user'].nunique()

        if totalUsers > 0:
            return float(contribUsers)/float(totalUsers)
        else:
            warnings.warn('contributingUsersHelper: total number of contributing users is zero')
            return None

    '''
    A wrapper function to calculate the averagae temporal user contribution counts within each community.
    Question #23
    Inputs: communities - Boolean to calculate the measurement seperately for each community (True) or for the full data set (False)
            unit - Time granularity for time series calculation
            eventType - List of event types to include
    Output: Dictionary of data frames for each community
    '''
    def getNumUserActions(self, communities=True,unit='D',eventType=None):

        if communities:
            return self.runCommunities(self.getNumUserActionsHelper,unit,eventType)
	else:
            return self.getNumUserActionsHelper(self.main_df,unit,eventType)

    '''
    Calculate the averagae temporal user contribution counts within the data set.
    Question #23
    Inputs: df - Events data frame
            unit - Time granularity for time series calculation
            eventType - List of event types to include
    Output: Data frame containing a time series of average event counts
    '''        
    def getNumUserActionsHelper(self,df,unit,eventType):
        
        if eventType != None:
            df = df[df.event.isin(eventType)]
        
        df['value'] = [0 for i in range(len(df))]
        df = df.set_index('time')
 
        #get event counts for each user within each time unit
        df = df[['user','value']].groupby([ pd.TimeGrouper(unit), 'user']).count()
        df = df.reset_index()

        #average the event counts across all users to get a single time series for the community
        df = df[['time','value']].groupby('time').mean()
        df['value'] = pd.to_numeric(df['value'])
        return df.reset_index()

    '''
    A wrapper function to calculate the burstiness of inter-event times within each community.
    Question #9
    Inputs: communities - Boolean to calculate the measurement seperately for each community (True) or for the full data set (False)
            eventType - List of event types to include in the data
    Output: Dictionary of burstiness values for each community
    '''    
    def burstsInCommunityEvents(self, communities=True,eventType = None):
        if communities:
            return self.runCommunities(self.burstsInCommunityEventsHelper,eventType)
        else:
            return self.burstsInCommunityEventsHelper(self.main_df,eventType)

    '''
    Calculates the burstiness of inter-event times within the data set.
    Question #9
    Inputs: communities - Boolean to calculate the measurement seperately for each community (True) or for the full data set (False)
            eventType - List of event types to include in the data
    Output: Burstiness value (scalar)
    '''            
    def burstsInCommunityEventsHelper(self, df,eventType):

        if eventType != None:
            df = df[df['event'].isin(eventType)]

        #get interevent times
        df['diff'] = df['time'].diff()
        df['diff'] = df['diff'] / np.timedelta64(1, 's')
        df = df[np.isfinite(df['diff'])]
        
        mean = df['diff'].mean()
        std = df['diff'].std()
        burstiness = (std - mean) / (std + mean)

        if not np.isnan(burstiness):
            return burstiness
        else:
            warnings.warn('burstsInCommunityEventsHelper: burstiness is NaN')
            return None

    '''
    A wrapper function to calculate the proportion of different issue action types as a function of time.
    Question #8 (Optional Measurement)
    Inputs: communities - Boolean to calculate the measurement seperately for each community (True) or for the full data set (False)
            unit - Temporal granularity for calculating the time series (e.g. D - day, H - hour, etc.)
    Output: Dictionary of data frames for each community
    '''    
    def propIssueEvent(self, communities=True,unit='D'):
        if communities:
            return self.runCommunities(self.propIssueEventHelper,unit)
        else:
            return self.propIssueEventHelper(self.main_df,unit)

    '''
    Calculates the proportion of different issue action types as a function of time.
    Question #8 (Optional Measurement)
    Inputs: communities - Boolean to calculate the measurement seperately for each community (True) or for the full data set (False)
            unit - Temporal granularity for calculating the time series (e.g. D - day, H - hour, etc.)
    Output: Dictionary of data frames for each community
    '''    
    def propIssueEventHelper(self,df,unit='D'):

        if self.main_df_opt is not None:            

            df = df[df['event'] == 'IssuesEvent']

            #round times down to nearest unit
            df = df.assign(time=df.time.dt.floor(unit))

            #merge optional columns (action, merged) with primary data frame
            df = df.merge(self.main_df_opt,how='left',left_index=True,right_index=True)

            df = df[['action','event','time']].groupby(['time','action']).count()  #time,action,count
            df = df.reset_index()

            p = df
            p.columns = ['time','action', 'counts']
            #create one column for each action type holding the counts of that action type
            p = p.pivot(index='time',columns='action', values='counts').fillna(0)
            p = p.reset_index()
            p = pd.melt(p, id_vars=['time'], value_vars=['closed', 'opened', 'reopened'])
            p = p.set_index('time')
            p.columns = ['action', 'value']

            return p.reset_index()
        else:
            warnings.warn('Skipping optional propIssueEventHelper')
            return None

    '''
    A wrapper function to calculate the distribution of user account ages for users who are active in each community.
    Question #10
    Inputs: communities - Boolean to calculate the measurement seperately for each community (True) or for the full data set (False)
            eventType - List of event types to include in the data
    Output: Dictionary of pandas Series containing account age distributions for each community
    '''        
    def ageOfAccounts(self, communities=True, eventType=None):
        if communities:
            return self.runCommunities(self.ageOfAccountsHelper,eventType)
        else:
            return self.ageOfAccountsHelper(self.main_df,eventType)

    '''
    Calculates the distribution of user account ages for users who are active in the data.
    Question #10
    Inputs: df - Events data 
                 eventType - List of event types to include in the data
    Output: A pandas Series containing account age of the user for each action taken in the community
    '''        
    def ageOfAccountsHelper(self,df,eventType):
        if self.useUserMetaData:
            if eventType != None:
                df  = df[df.event.isin(eventType)]

            df = df.merge(self.created_at_df, left_on='user', right_on='user', how='inner')
            df = df.sort_values(['time'])

            #get user account age at the time of each event
            df['age'] = df['time'].sub(df['created_at'], axis=0)
            df['age'] = df['age'].astype('timedelta64[D]')
            return df['age']
        else:
            warnings.warn('Skipping ageOfAccountsHelper because metadata file is required')
            return None


    '''
    A wrapper function to calculate the distribution of user geolocations for users active in each community
    Question #21
    Inputs: communities - Boolean to calculate the measurement seperately for each community (True) or for the full data set (False)
            eventType - List of event types to include in the data
    Output: Dictionary of data frames with the location distribution for each community
    '''            
    def userGeoLocation(self, communities=True, eventType=None):
        if communities:
            return self.runCommunities(self.userGeoLocationHelper,eventType)
        else:
            return self.userGeoLocationHelper(self.main_df,eventType)

    '''
    A wrapper function to calculate the distribution of user geolocations for users active in each community
    Question #21
    Inputs: df - Events data frame 
            eventType - List of event types to include in the data
    Output: Data frame with the location distribution of activity in the data
    '''            
    def userGeoLocationHelper(self,df,eventType):

        if not eventType is None:
            df = df[df.event.isin(eventType)]

        if self.useUserMetaData:

            #merge events data with user location metadata
            merge = df.merge(self.locations_df, left_on='user', right_on='user', how='inner')
            merge = merge[['user','location']].groupby(['location']).count()
            merge.columns = ['value']
            merge = merge.sort_values('value',ascending=False).reset_index()

            #set rare locations to "other"
            thresh = 0.007*merge.value.sum()
            merge['location'][merge['value'] < thresh] = 'other'
            
            grouped = merge.groupby('location').sum()

            return grouped.reset_index()
        else:
            warnings.warn('Skipping userGeoLocationHelper because metadata file is required')
            return None
    
    '''
    Wrapper function to calculate the distribution of user inter-event time burstiness within each community.
    Question #9
    Inputs: communities - Boolean to calculate the measurement seperately for each community (True) or for the full data set (False)
            eventType - List of event types to include in the data
            thresh - Minimum number of events for a user to be included
    Output: Dictionary of data frames for each community
    '''            
    def getUserBurstByCommunity(self, communities=True,eventType = None, thresh=10):
        if communities:
            return self.runCommunities(self.getUserBurstByCommunityHelper,eventType,thresh)
        else:
            return self.getUserBurstByCommunityHelper(self.main_df,eventType,thresh)

    '''
    Calculate the distribution of user inter-event time burstiness in the data set.
    Question #9
    Inputs: df - Events data frame
            eventType - List of event types to include in the data
            thresh - Minimum number of events for a user to be included
    Output: Data frame of user burstiness values
    '''            
    def getUserBurstByCommunityHelper(self,df,eventType,thresh):
        
        if eventType != None:
            df = df[df.event.isin(eventType)]

        #only calculate burstiness for users which have sufficient activity
        users = df.groupby('user')
        user_counts = users['event'].count().reset_index()
        user_list = user_counts['user'][user_counts['event'] >= thresh]       
        df = df[df['user'].isin(user_list)]


        if len(df.index) > 0:
            #get interevent times for each user seperately
            if len(df['user'].unique()) > 1:
                measurement = df.groupby('user').apply(lambda grp: (grp.time - grp.time.shift()).fillna(0)).reset_index()
            else:
                df['time'] = df['time'] - df['time'].shift().dropna()
                measurement = df.copy()
                
            measurement['value'] = measurement['time'] / np.timedelta64(1, 's')
        
            #calculate burstiness using mean and standard deviation of interevent times
            measurement = measurement.groupby('user').agg({'value':{'std':np.std,'mean':np.mean}})
            
            measurement.columns = measurement.columns.get_level_values(1)
            measurement['burstiness'] = (measurement['std'] - measurement['mean']) / (measurement['std'] + measurement['mean'])
            

            measurement = measurement[['burstiness']].dropna()
            measurement = measurement.reset_index()
            del measurement['user']

            return measurement['burstiness']
        else:
            warnings.warn('getUserBurstByCommunityHelper: Not enough active users')
            return None

    '''
    Wrapper function calculate the gini coefficient for the data frame.
    Question #6
    Input: communities - Boolean to calculate the measurement seperately for each community (True) or for the full data set (False) 
           eventType - A list of event types to include in the calculation
    Output: A dictionary of gini coefficients for each community
    '''
    def getCommunityGini(self,communities=True,eventType=None):
        if communities:
            return self.runCommunities(self.getGiniCoefHelper,'repo',eventType)
        else:
            return self.getGiniCoefHelper(self.main_df,nodeType='repo',eventType=eventType)

    '''
    Wrapper function calculate the Palma coefficient for the data frame.
    Question #6
    Input: communities - Boolean to calculate the measurement seperately for each community (True) or for the full data set (False) 
           eventType - A list of event types to include in the calculation
    Output: A dictionary of Palma coefficients for each community
    '''
    def getCommunityPalma(self,communities=True,eventType=None):
        if communities:
            return self.runCommunities(self.getPalmaCoefHelper,'repo',eventType)
        else:
            return self.getPalmaCoefHelper(self.main_df,nodeType='repo',eventType=eventType)



    
