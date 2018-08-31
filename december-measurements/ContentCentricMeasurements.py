from __future__ import division
import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import Pool
from functools import partial
from pathos import pools as pp
from multiprocessing import Pool
import pickle
from validators import check_empty

'''
This class implements content centric methods.
These metrics assume that the data is in the order created_at,type,actor.id,content.id
'''


class ContentCentricMeasurements(object):
    def __init__(self):
        super(ContentCentricMeasurements, self).__init__()


    def getNodeDictionary(self,df):

        meas = {}
        for content in df.content.unique():
            meas[content] = df[df.content == content]
            del meas[content]["content"]

        return meas

    def getSelectContentIds(self, content_ids):

        '''
        This function creates a dictionary of data frames with
        each entry being the activity of one piece of content from the content_ids
        argument.

        This is used for the selected content ids for the node-level meausurements.
        Inputs: content_ids - List of content ids (e.g. GitHub - full_name_h, etc.)
        Output: Dictionary of data frames with the content ids as the keys
        '''


        contentDic = {}
        for ele in content_ids:
            d = self.main_df[self.main_df['content'] == ele]
            contentDic[ele] = d
        return contentDic


    def runSelectContentIds(self, method, *args):

        '''
        This function runs a particular measurement (method) on the
        content ids that were selected by getSelectContentIds.

        This is used for the selected content IDs for the node-level meausurements.

        Inputs: method - Measurement function
        Output: Dictionary of measurement results with the content ids as the keys
        '''

        ans = {}
        for ele in self.selectedContent.keys():
            df = self.selectedContent[ele].copy()
            ans[ele] = method(df,*args)
        return ans


    def getContentDiffusionDelay(self, eventTypes=None, selectedContent=True, time_bin='m',content_field='root'):

        '''
        This method returns the distributon for the diffusion delay for each content node.
        Question #1
        Inputs: DataFrame - Data
            eventTypes - A list of events to filter data on
            selectedContent - A boolean indicating whether to run on selected content nodes
            time_bin - Time unit for time differences, e.g. "s","d","h"
        Output: An dictionary with a data frame for each content ID containing the diffusion delay values in the given units
        '''

        df = self.selectedContent.copy()

        if not eventTypes is None:
            df = df[df.event.isin(eventTypes)]

        if len(df.index) == 0: 
            return {}


        #use metadata for content creation dates if available
        if self.useContentMetaData:
            df = df.merge(self.contentMetaData,left_on=content_field,right_on=content_field,how='left')
            df = df[[content_field,'created_at','time']].dropna()
            df['value'] = (df['time']-df['created_at']).apply(lambda x: int(x / np.timedelta64(1, time_bin)))
        #otherwise use first observed activity as a proxy
        else:
            creation_day = df.groupby(content_field)['time'].min().reset_index()
            creation_day.columns = [content_field,'creation_date']
            df = df.merge(creation_day, on=content_field, how='left')
            df['value'] = (df['time']-df['creation_date']).apply(lambda x: int(x / np.timedelta64(1, time_bin)))
            df = df[[content_field,'value']]
            df.columns = ['content','value']
            df = df.iloc[1:]


        measurements = self.getNodeDictionary(df)

        return measurements


    def getContentGrowth(self, eventTypes=None, cumSum=False, time_bin='D', content_field='root'):

        '''
        This method returns the growth of a repo over time.
        Question #2
        Input:   eventTypes - A list of events to filter data on
                 cumSum - This is a boolean that indicates if the dataframe should be cumuluative over time.
                 time_bin - The temporal granularity of the output time series
        output - A dictionary with a dataframe for each content id that describes the content activity growth. 
        '''


        df = self.selectedContent

        if not eventTypes is None:
            df = df[df.event.isin(eventTypes)]

        df = df.set_index("time")

        measurement = df[[content_field,'event']].groupby([content_field,pd.Grouper(freq=time_bin)]).count()
        measurement.columns = ['value']

        if cumSum == True:
            measurement['value'] = measurement.cumsum(axis=0)['value']
        measurement = measurement.reset_index()
        measurement.columns = ['content','time','value']

        measurements = self.getNodeDictionary(measurement)

        return measurements


    def getContributions(self, new_users_flag=False,cumulative=False,eventTypes=None,time_bin='H',content_field="root"):

        '''
        Calculates the total number of unique daily contributers to a repo or the unique daily contributors who are new contributors
        Question # 4
            Input: newUsersOnly - Boolean to indicate whether to calculate total daily unique users (False) or daily new contributers (True),
                                  if None run both total and new unique users.
                   cumulative - Boolean to indicate whether or not the metric should be cumulative over time
                   eventTypes - A list of event types to include in the calculation
                   time_bin - Granularity of time series
            Output: A data frame with daily event counts
        '''

        df = self.selectedContent.copy()


        def contributionsInsideHelper(dfH,newUsersOnly,cumulative):
            if newUsersOnly:
                #drop duplicates on user so a new user only shows up once in the data
                dfH = dfH.drop_duplicates(subset=['user'])

            p = dfH[[content_field,'user']].groupby([content_field,pd.Grouper(freq=time_bin)])['user'].nunique().reset_index()
            
            if cumulative:
                #get cumulative user counts
                p['user'] = p.groupby(content_field)['user'].transform(pd.Series.cumsum)
            
            p.columns = ['content','time','value']
            return p

        if eventTypes != None:
            df = df[df.event.isin(eventTypes)]

        df = df.set_index("time")

        if not new_users_flag:
            #run total daily user counts
            results = contributionsInsideHelper(df,False, cumulative)
           

        else:
            #run unique daily user counts
            results = contributionsInsideHelper(df,newUsersOnly, cumulative)
           
        meas = self.getNodeDictionary(results)
            
        return meas


    def getDistributionOfEvents(self,weekday=False,content_field="root"):

        '''
          This method returns the distribution for each event over time or by weekday. Default is over time. 
          Question #5
          Inputs: weekday - (Optional) Boolean to indicate whether the distribution should be done by weekday. Default is False.
          Output: Dataframe with the distribution of events by weekday. Columns: Event, Weekday, Count or Event, Date, Count
        '''


        df = self.selectedContent.copy()

        df['id'] = df.index
        df['weekday'] = df['time'].dt.weekday_name
        df['date'] = df['time'].dt.date

        if weekday:
            col = 'weekday'
        else:
            col = 'date'

        counts = df.groupby([content_field,'event',col])['user'].count().reset_index()
        counts.columns = ['content','event',col,'value']

        meas = self.getNodeDictionary(counts)
            
        return meas


    def processDistOfEvents(self,df,weekday):
        '''
           Helper Function for getting the Dist. of Events per weekday.

        '''

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



    @check_empty(default=None)
    def getGiniCoef(self,nodeType='root', eventTypes=None, content_field="root"):
        '''
        Wrapper function calculate the gini coefficient for the data frame.
        Question #6,14,26
        Input: df - Data frame containing data can be any subset of data
               nodeType - Type of node to calculate the Gini coefficient over.  Options: user or repo (case sensitive)
               eventTypes - A list of event types to include in the calculation
        Output: g - gini coefficient
        '''

        return self.getGiniCoefHelper(self.main_df, nodeType, eventTypes, content_field)


    def getGiniCoefHelper(self, df,nodeType,eventTypes=None,content_field="root"):

        '''
        This method returns the gini coefficient for the data frame.
        Question #6,14,26
        Input: df - Data frame containing data can be any subset of data
               nodeType - Type of node to calculate the Gini coefficient over.  Options: user or repo (case sensitive)
               eventTypes - A list of event types to include in the calculation
        Output: g - gini coefficient
        '''


        if eventTypes is not None:
            df = df[df.event.isin(eventTypes)]


        #count events for given node type
        if nodeType != 'user':
            df = df[[nodeType, 'user']].groupby(nodeType).count()
        else:
            df = df[[nodeType, content_field]].groupby(nodeType).count()

        df.columns = ['value']
        df = df.reset_index()

        values = df['value'].values.astype(float)

        if np.amin(values) < 0:
            values -= np.amin(values) 
 
        values += 1e-9

        values = np.sort(np.array(values))

        index = np.arange(1,values.shape[0]+1)
        n = values.shape[0]
        g = ((np.sum((2 * index - n  - 1) * values)) / (n * np.sum(values))) 
        
        return g


    def getPalmaCoef(self,nodeType='root', eventTypes=None, content_field="root"):
        '''
        Wrapper function calculate the Palma coefficient for the data frame.
        Question #6,14,26
        Input: df - Data frame containing data can be any subset of data
               nodeType - Type of node to calculate the Palma coefficient over.  Options: user or repo (case sensitive)
               eventTypes - A list of event types to include in the calculation
        Output: Palma coefficient
        '''

        return self.getPalmaCoefHelper(self.main_df, nodeType,eventTypes,content_field)


    @check_empty(default=None)
    def getPalmaCoefHelper(self, df, nodeType='root', eventTypes=None, content_field = "root"):

        '''
        This method returns the Palma coefficient.
        Question #6,14,26
        Input: df - Data frame containing data can be any subset of data
               nodeType - (Optional) This is the node type on whose event counts the Palma coefficient
                          is calculated. Options: user or content (case sensitive)
               eventTypes - A list of event types to include in the calculation
        Output: p - Palma Coefficient
        '''

        
        if eventTypes is not None:
            df = df[df.event.isin(eventTypes)]

        if nodeType != 'user':
            df = df[[nodeType, 'user']].groupby(nodeType).count()
        else:
            df = df[[nodeType, content_field]].groupby(nodeType).count()

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



    def getTopKContent(self,content_field='root',k=100,eventTypes=None):

        '''
        This method returns the top-k pieces of content by event count for selected event types
        Question #12,13
        Inputs: eventTypes - A list of event types to include in the calculation
                content_field - Options: root, parent, or content.
                k - Number of entities to return
        Outputs: Dataframe with the top-k content ids and their event counts. Columns are content id and the count of that event.
        '''


        df = self.main_df

        if not eventTypes is None:
            df = df[df.event.isin(eventTypes)]
        p = df[[content_field, 'event']].groupby([content_field]).count()
        p = p.sort_values(by='event',ascending=False)
        p.columns = ['value']
        return p.head(k)



    def getDistributionOfEventsByContent(self,content_field='root',eventTypes=['WatchEvent']):

        '''
        This method returns the distribution of event type per content e.g. x repos/posts/tweets with y number of events, 
        z repos/posts/ with n amounts of events.
        Question #11,12,13
        Inputs: eventTypes - List of event type(s) to get distribution over
        Outputs: Dataframe with the distribution of event type per repo. Columns are repo id and the count of that event.
        '''


        df = self.main_df

        if eventTypes != None:
            df = df[df['event'].isin(eventTypes)]

        p = df[[content_field,'time']].groupby(content_field).count()
        p = p.sort_values(by='time')
        p.columns = ['value']
        p = p.reset_index()
        return p


    def getRepoPullRequestAcceptance(self,eventTypes=['PullRequestEvent'],thresh=2):

        '''
        Calculate the proportion of pull requests that are accepted for each repo.
        Question #15 (Optional Measurement)
        Inputs: eventTypes: List of event types to include in the calculation (Should be PullRequestEvent).
                thresh: Minimum number of PullRequests a repo must have to be included in the distribution.
        Output: Data frame with the proportion of accepted pull requests for each repo
        '''


        df = self.main_df_opt

        #check if optional columns exist
        if not df is None and 'PullRequestEvent' in self.main_df.event.values:
            df = df[self.main_df.event.isin(eventTypes)]
            users_repos = self.main_df[self.main_df.event.isin(eventTypes)]

            #subset to only pull requests which are being closed (not opened)
            idx = df['action'] == 'closed'
            closes = df[idx]
            users_repos = users_repos[idx]

            #merge optional columns (action, merged) with the main data frame columns
            closes = pd.concat([users_repos,closes],axis=1)
            closes = closes[['content','merged']]
            closes['value'] = 1

            #create count of accepted (merged) and rejected pull requests by repo
            outcomes = closes.pivot_table(index=['content'],values=['value'],columns=['merged'],aggfunc='sum').fillna(0)
            
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

            #subset on content which have enough data
            outcomes = outcomes[outcomes['total'] >= thresh]

            if len(outcomes.index) > 0:
                measurement = outcomes.reset_index()[['content','value']]
            else:
                measurement = None
        else:
            measurement = None


        return measurement


    def getEventTypeRatioTimeline(self,eventTypes=None,event1='IssuesEvent',event2='PushEvent',content_field="root"):

        if self.platform != 'reddit':
            df = self.selectedContent.copy()
        else:
            df = self.main_df.copy()

        if eventTypes != None:
            df = df[df['event'].isin(eventTypes)]

        df['value'] = 1

        if len(df.index) < 1:
            return {}

        measurement = df.groupby([content_field,'user']).apply(lambda x: x.value.cumsum()).reset_index()
        measurement['event'] = df['event'].reset_index(drop=True)


        if self.previous_event_counts is not None:
            measurement = measurement.merge(self.previous_event_counts,on=['user',content_field],how='left').fillna(0)
            measurement['value'] = measurement['value'] + measurement['count']

        measurement = measurement[measurement['event'].isin([event1,event2])]

        measurement[event1] = measurement['event'] == event1
        measurement[event2] = measurement['event'] == event2


        measurement['next_event_' + event1] = measurement[event1].shift(-1)
        measurement['next_event_' + event2 ] = measurement[event2].shift(-1)

        bins = np.logspace(-1,3.0,16)
        measurement['num_events_binned'] = pd.cut(measurement['value'],bins).apply(lambda x: np.floor(x.right)).astype(float)


        def ratio(grp):

            if float(grp['next_event_' + event2].sum()) > 0:
                return float(grp['next_event_' + event1].sum()) / float(grp['next_event_' + event2].sum())
            else:
                return 0.0

        if len(measurement.index) > 0:
            measurement = measurement.groupby([content_field,'num_events_binned']).apply(ratio).reset_index()
            measurement.columns = ['content','num_events_binned','value']
        else:
            measurement = None

        measurement = self.getNodeDictionary(measurement)

        return(measurement)


    def propUserContinue(self,eventTypes=None,content_field="root"):
        
        if self.platform != 'reddit':
            df = self.selectedContent.copy()
        else:
            df = self.main_df.copy()
        
        if not eventTypes is None:            
            data = df[df['event'].isin(eventTypes)]

        if len(data.index) > 1:
            data['value'] = 1
            grouped = data.groupby(['user',content_field])

            #get running count of user actions on each piece of content
            if grouped.ngroups > 1:
                measurement = grouped.apply(lambda grp: grp.value.cumsum()).reset_index()
            else:
                data['value'] = data['value'].cumsum()
                measurement = data.copy()

            #get total number of user actions on each piece of content
            grouped = measurement.groupby(['user',content_field]).value.max().reset_index()
            grouped.columns = ['user',content_field,'num_events']

            measurement = measurement.merge(grouped,on=['user',content_field])

            #boolean indicator of whether a given event is the last one by the user
            measurement['last_event'] = measurement['value'] == measurement['num_events']

            #add event counts from before the start of the test period
            if self.previous_event_counts is not None:
                measurement = measurement.merge(self.previous_event_counts,on=['user',content_field],how='left').fillna(0)
                measurement['value'] = measurement['value'] + measurement['count']


            #bin by the number of previous events
            bins = np.logspace(-1,2.5,30)
            measurement['num_actions'] = pd.cut(measurement['value'],bins).apply(lambda x: np.floor(x.right)).astype(float)

            measurement['last_event'] = ~measurement['last_event']


            #get percentage of events within bin that are NOT the last event for a user
            measurement = measurement.groupby([content_field,'num_actions']).last_event.mean().reset_index()
            measurement.columns = ['content','num_actions','value']

            measurement = self.getNodeDictionary(measurement)

        else:
            measurement = {}

        return measurement
