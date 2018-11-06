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
            try:
                self.locations_df = self.userMetaData[['user','city','country']]
            except:
                self.locations_df = self.userMetaData[['user','location']]
            
    '''
    This method loads the community dictionary from the specified pickle file.
    The pickle file should contain a dictionary of the format
    {"type_of_community1":{"community1":[repo1,repo2,...], "community2":[repo3,repo4,...]},
     "type_of_community2":...}
     Inputs: path - file path of pickle file
    '''
    def loadCommunities(self,path,content_field='content'):
        if path != '':
            with open(path, 'rb') as handle:
                self.comDic = pkl.load(handle)
                for key in self.comDic.keys():
                    print(self.comDic[key].keys())
        else:
            self.comDic = {"topic":{"all":self.main_df[content_field].unique()}}


    def getCommmunityDF(self,community_col='subreddit'):

        if community_col in self.main_df.columns:
            return self.main_df.copy()
        elif community_col != '':
            self.loadMetaData() 
            self.loadCommunities(self.community_dict_file)
     
       
            dfs = []
     
            content_community_types = ['topic',"languages"]
            user_community_types = ['city','country','company',"locations"]
        
            #content-focused communities
            for community in content_community_types:
                if community in self.comDic.keys():
                    for key in self.comDic[community]:
                        d = self.main_df[self.main_df['content'].isin(self.comDic[community][key])]
                        d[community_col] = key
                        dfs.append(d)

            #user-focused communities
            for community in user_community_types:
                if community in self.comDic.keys():
                    for key in self.comDic[community]:
                        d = self.main_df[self.main_df['user'].isin(self.comDic[community][key])]
                        d[community_col] = key
                        dfs.append(d)     

            return pd.concat(dfs)
                    

    def getCommunityMeasurementDict(self,df):

        meas = {}
        if isinstance(df,pd.DataFrame):
            for community in df.community.unique():
                meas[community] = df[df.community == community]
                del meas[community]["community"]
        elif isinstance(df,pd.Series):
            series_output = False
            for community in df.index:
                meas[community] = df[community]
                try:
                    len(df[community])
                    series_output = True
                except:
                    ''
            if series_output:
                for community in meas:
                    meas[community] = pd.Series(meas[community])

        return meas

   
    '''
    Calculates the proportion of each event type in the data.
    Question #7
    Inputs: df - Events data frame
            communities - Boolean to calculate the measurement seperately for each community (True) or for the full data set (False)
            eventTypes - List of event types to include
    Output: Dictionary of data frames with columns for event type and proportion, with one data frame for each community
    '''
    def getProportion(self, eventTypes=None, community_field="subreddit"):

        df = self.communityDF.copy()

        if eventTypes != None:
            df = df[df['event'].isin(eventTypes)]

        p = df[['user','event',community_field]].groupby([community_field,'event']).count()
        p = p.reset_index()
        p.columns = ['community','event', 'value']

        community_totals = p[['community','value']].groupby('community').sum().reset_index()
        community_totals.columns = ['community','total']
        p = p.merge(community_totals, on='community',how='left')

        p['value'] = p['value']/p['total']
        del p['total']

        measurement = self.getCommunityMeasurementDict(p)

        return measurement


    '''
    This method calculates the proportion of users with events in teh data who are active contributors.
    Question #20
    Inputs: df - Events data
    Output: Proportion of all users who make active contributions
    '''
    def contributingUsers(self,eventTypes=None,community_field="subreddit"):

        df = self.communityDF.copy()

        if not eventTypes is None:
            df = df[df.event.isin(eventTypes)]

        #total number of unique users
        totalUsers = df.groupby(community_field)['user'].nunique()
        totalUsers.name = 'total_users'

        df = df[df['event'].isin(self.contribution_events)]

        #number of unique users with direct contributions
        contribUsers = df.groupby(community_field)['user'].nunique()
        contribUsers.name = 'contributing_users'

        df = pd.concat([totalUsers, contribUsers], axis=1).fillna(0)

        df['value'] = df['contributing_users']/ df['total_users']

        measurement = self.getCommunityMeasurementDict(df['value'])

        return measurement


    '''
    Calculate the average temporal user contribution counts within the data set.
    Question #23
    Inputs: df - Events data frame
            unit - Time granularity for time series calculation
            eventTypes - List of event types to include
    Output: Data frame containing a time series of average event counts
    '''        
    def getNumUserActions(self,unit='h',eventTypes=None,community_field='subreddit'):
        
        df = self.communityDF.copy()

        if eventTypes != None:
            df = df[df.event.isin(eventTypes)]
        
        df['value'] = [0 for i in range(len(df))]
        df = df.set_index('time')

        #get event counts for each user within each time unit
        df = df[['user','value',community_field]].groupby([ pd.TimeGrouper(unit), 'user', community_field]).count()
        df = df.reset_index()

        #average the event counts across all users to get a single time series for the community
        df = df[['time','value',community_field]].groupby(['time',community_field]).mean().reset_index()
        df['value'] = pd.to_numeric(df['value'])
        df.columns = ['time','community','value']

        measurement = self.getCommunityMeasurementDict(df)
        

        return measurement


    '''
    Calculates the burstiness of inter-event times within the data set.
    Question #9
    Inputs: communities - Boolean to calculate the measurement seperately for each community (True) or for the full data set (False)
            eventTypes - List of event types to include in the data
    Output: Burstiness value (scalar)
    '''            
    def burstsInCommunityEvents(self,eventTypes=None,community_field="subreddit"):

        df = self.communityDF.copy()

        if eventTypes != None:
            df = df[df['event'].isin(eventTypes)]

        def burstiness(grp):

            #get interevent times
            grp['diff'] = grp['time'].diff()
            grp['diff'] = grp['diff'] / np.timedelta64(1, 's')
 
            grp = grp[np.isfinite(grp['diff'])]
                        
            mean = grp['diff'].mean()
            std = grp['diff'].std()
            if std + mean > 0:
                burstiness = (std - mean) / (std + mean)
            else:
                burstiness = 0

            return burstiness

        b = df.groupby(community_field).apply(burstiness)
        b.columns = ['community','value']

        measurement = self.getCommunityMeasurementDict(b)

        return measurement


    '''
    Calculates the proportion of different issue action types as a function of time.
    Question #8 (Optional Measurement)
    Inputs: communities - Boolean to calculate the measurement seperately for each community (True) or for the full data set (False)
            unit - Temporal granularity for calculating the time series (e.g. D - day, H - hour, etc.)
    Output: Dictionary of data frames for each community
    '''    
    def propIssueEvent(self,unit='D'):

        df = self.communityDF.copy()

        if self.main_df_opt is not None:            

            df = df[ (df['event'] == 'IssuesEvent') ]

            if len(df) == 0:
                return(None)

            #round times down to nearest unit
            df = df.assign(time=df.time.dt.floor(unit))

            #merge optional columns (action, merged) with primary data frame
            df = df.merge(self.main_df_opt,how='left',left_index=True,right_index=True)
            
            df = df[(df['action'].isin(['opened','closed','reopened']))]

            if len(df) == 0:
                return(None)

            df = df[['action','event','time','community']].groupby(['time','action','community']).count()  #time,action,count
            df = df.reset_index()


            p = df
            p.columns = ['time','action', 'community','counts']

            #create one column for each action type holding the counts of that action type
            p = pd.pivot_table(p,index=['time','community'],columns='action', values='counts').fillna(0)
            p = p.reset_index()
            p = pd.melt(p, id_vars=['time','community'], value_vars=['closed', 'opened', 'reopened'])
            p.columns = ['time','community','action', 'value']

            measurement = self.getCommunityMeasurementDict(p)

            return measurement
        else:
            warnings.warn('Skipping optional propIssueEventHelper')
            return None


    '''
    Calculates the distribution of user account ages for users who are active in the data.
    Question #10
    Inputs: df - Events data 
                 eventTypes - List of event types to include in the data
    Output: A pandas Series containing account age of the user for each action taken in the community
    '''        
    def ageOfAccounts(self,eventTypes=None,community_field="subreddit"):
        
        df = self.communityDF.copy()
        
        if self.useUserMetaData:

            if eventTypes != None:
                df  = df[df.event.isin(eventTypes)]


            df = df.merge(self.created_at_df, left_on='user', right_on='user', how='inner')
            df = df.sort_values(['time'])

            #get user account age at the time of each event
            df['age'] = df['time'].sub(df['created_at'], axis=0)
            df['age'] = df['age'].astype('timedelta64[D]')

            df = df.rename(index=str, columns={community_field: "community"})
            df.set_index('community')

            measurement = self.getCommunityMeasurementDict(df['age'])

            return df['age']
        else:
            warnings.warn('Skipping ageOfAccountsHelper because metadata file is required')
            return {}


    '''
    A function to calculate the distribution of user geolocations for users active in each community
    Question #21
    Inputs: df - Events data frame 
            eventTypes - List of event types to include in the data
    Output: Data frame with the location distribution of activity in the data
    '''            
    def userGeoLocation(self,eventTypes=None,community_field="subreddit"):

        df = self.communityDF.copy()

        if not eventTypes is None:
            df = df[df.event.isin(eventTypes)]

        if self.useUserMetaData:

            #merge events data with user location metadata
            merge = df.merge(self.locations_df, left_on='user', right_on='user', how='inner')
            merge = merge[['user','country',community_field]].groupby([community_field,'country']).count().reset_index()
            merge.columns = ['community','country','value']

            community_totals = merge.groupby(community_field)['value'].sum().reset_index()
            community_totals.columns = ['community','total']
            merge = merge.merge(community_totals,on='community',how='left')
            merge['proportion'] /= merge['total']


            #set rare locations to "other"
            thresh = 0.007
            merge['country'][merge['proportion'] < thresh] = 'other'
            
            #sum over other countries
            grouped = merge.groupby('country').sum().reset_index()

            measurement = self.getCommunityMeasurementDict(grouped)

            return measurement
        else:
            warnings.warn('Skipping userGeoLocationHelper because metadata file is required')
            return {}
    

    '''
    Calculate the distribution of user inter-event time burstiness in the data set.
    Question #9
    Inputs: df - Events data frame
            eventTypes - List of event types to include in the data
            thresh - Minimum number of events for a user to be included
    Output: Data frame of user burstiness values
    '''            
    def getUserBurstByCommunity(self,eventTypes=None,thresh=5.0,community_field="subreddit"):
        
        df = self.communityDF.copy()

        if eventTypes != None:
            df = df[df.event.isin(eventTypes)]

        #only calculate burstiness for users which have sufficient activity
        users = df.groupby(['user',community_field])
        user_counts = users['event'].count().reset_index()
        user_list = user_counts[user_counts['event'] >= thresh]
        user_list.columns = ['user',community_field,'total_activity']

        if len(user_list) == 0:
            return None

        df = df.merge(user_list,how='inner',on=['user',community_field])
      
        def user_burstiness(grp):
            #get interevent times for each user seperately
            if len(grp['user'].unique()) > 1:
                grp = grp.groupby('user').apply(lambda grp: (grp.time - grp.time.shift()).fillna(0)).reset_index()
            else:
                grp['time'] = grp['time'] - grp['time'].shift().dropna()
                
            grp['value'] = grp['time'] / np.timedelta64(1, 's')
        
            #calculate burstiness using mean and standard deviation of interevent times
            grp = grp.groupby('user').agg({'value':{'std':np.std,'mean':np.mean}})
            
            grp.columns = grp.columns.get_level_values(1)
            grp['value'] = (grp['std'] - grp['mean']) / (grp['std'] + grp['mean'])
            

            grp = grp[['value']].dropna()
            grp = grp.reset_index()

            return grp

        b = df.groupby(community_field).apply(user_burstiness).reset_index()[[community_field,'value']].set_index(community_field)['value']

        measurement = self.getCommunityMeasurementDict(b)

        return measurement

    '''
    Wrapper function calculate the gini coefficient for the data frame.
    Question #6
    Input: communities - Boolean to calculate the measurement seperately for each community (True) or for the full data set (False) 
           eventTypes - A list of event types to include in the calculation
    Output: A dictionary of gini coefficients for each community
    '''
    def getCommunityGini(self,communities=True,eventTypes=None,community_field="subreddit",content_field="root"):

        ginis = self.communityDF.groupby(community_field).apply(lambda x: self.getGiniCoefHelper(x,content_field))
        
        measurement = self.getCommunityMeasurementDict(ginis)

        return measurement

    '''
    Wrapper function calculate the Palma coefficient for the data frame.
    Question #6
    Input: communities - Boolean to calculate the measurement seperately for each community (True) or for the full data set (False) 
           eventTypes - A list of event types to include in the calculation
    Output: A dictionary of Palma coefficients for each community
    '''
    def getCommunityPalma(self,communities=True,eventTypes=None,community_field="subreddit",content_field="root"):

        palmas = self.communityDF.groupby(community_field).apply(lambda x: self.getPalmaCoefHelper(x,content_field))

        measurement = self.getCommunityMeasurementDict(palmas)

        return measurement



    
