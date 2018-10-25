import pandas as pd
import numpy  as np

from datetime        import datetime
from multiprocessing import Pool
from functools       import partial
from pathos          import pools as pp

import pickle as pkl

from UserCentricMeasurements      import *
from ContentCentricMeasurements   import *
from CommunityCentricMeasurements import *

from TEMeasurements import *
from collections    import defaultdict

import jpype
import json
import os

basedir = os.path.dirname(__file__)

class BaselineMeasurements(UserCentricMeasurements, ContentCentricMeasurements, TEMeasurements, CommunityCentricMeasurements):
    def __init__(self,
                 dfLoc,
                 content_node_ids=[],
                 user_node_ids=[],
                 metaContentData=False,
                 metaUserData=False,
                 contentActorsFile=os.path.join(basedir, './baseline_challenge_data/filtUsers-baseline.pkl'),
                 contentFile=os.path.join(basedir, './baseline_challenge_data/filtRepos-baseline.pkl'),
                 topNodes=[],
                 topEdges=[],
                 previousActionsFile='',
                 community_dictionary='',
#                 community_dictionary=os.path.join(basedir, './baseline_challenge_data/baseline_challenge_community_dict.pkl'),
                 te_config=os.path.join(basedir, './baseline_challenge_data/te_params_baseline.json'),
                 platform='github',
                 use_java=True):
        super(BaselineMeasurements, self).__init__()

        self.platform = platform

        try:
            # check if input is a data frame
            dfLoc.columns
            df = dfLoc
        except:
            # if not it should be a csv file path
            df = pd.read_csv(dfLoc)

        self.contribution_events = ['PullRequestEvent',
                                    'PushEvent',
                                    'IssuesEvent',
                                    'IssueCommentEvent',
                                    'PullRequestReviewCommentEvent',
                                    'CommitCommentEvent',
                                    'CreateEvent',
                                    'post',
                                    'tweet']

        self.popularity_events = ['WatchEvent',
                                  'ForkEvent',
                                  'comment',
                                  'post',
                                  'retweet',
                                  'quote',
                                  'reply']

        print('preprocessing...')

        self.main_df = self.preprocess(df)

        print('splitting optional columns...')

        # store action and merged columns in a seperate data frame that is not used for most measurements
        if platform == 'github' and len(self.main_df.columns) == 6:
            self.main_df_opt = self.main_df.copy()[['action', 'merged']]
            self.main_df = self.main_df.drop(['action', 'merged'], axis=1)
        else:
            self.main_df_opt = None

        # For content centric
        print('getting selected content IDs...')

        if self.platform == 'reddit':
            self.selectedContent = self.main_df[self.main_df.root.isin(content_node_ids)]
        elif self.platform == 'twitter':
            self.selectedContent = self.main_df[self.main_df.parent.isin(content_node_ids)]
        else:
            self.selectedContent = self.main_df[self.main_df.content.isin(content_node_ids)]

        # For userCentric
        self.selectedUsers = self.main_df[self.main_df.user.isin(user_node_ids)]

        print('processing repo metatdata...')

        # read in external metadata files
        # repoMetaData format - full_name_h,created_at,owner.login_h,language
        # userMetaData format - login_h,created_at,location,company

        if metaContentData != False:
            self.useContentMetaData = True
            meta_content_data = pd.read_csv(metaContentData)
            self.contentMetaData = self.preprocessContentMeta(meta_content_data)
        else:
            self.useContentMetaData = False
        print('processing user metatdata...')
        if metaUserData != False:
            self.useUserMetaData = True
            self.userMetaData = self.preprocessUserMeta(pd.read_csv(metaUserData))
        else:
            self.useUserMetaData = False

        # For Community
        self.community_dict_file = community_dictionary
        print('getting communities...')
        if self.platform == 'github':
            self.communityDF = self.getCommmunityDF(community_col='community')
        elif self.platform == 'reddit':
            self.communityDF = self.getCommmunityDF(community_col='subreddit')
        else:
            self.communityDF = self.getCommmunityDF(community_col='')

        # read in previous events count external file (used only for one measurement)
        try:
            print('reading previous counts...')
            self.previous_event_counts = pd.read_csv(previousActionsFile)
        except:
            self.previous_event_counts = None

        print('previous event counts', self.previous_event_counts)

        # For TE
        if use_java:
            print('starting jvm...')
            if not jpype.isJVMStarted():
                jpype.startJVM(jpype.getDefaultJVMPath(),
                               '-ea',
                               '-Djava.class.path=infodynamics.jar')

        # read pkl files which define nodes of interest for TE measurements
        self.repo_actors = self.readPickleFile(contentActorsFile)
        self.repo_groups = self.readPickleFile(contentFile)

        self.top_users = topNodes
        self.top_edges = topEdges

        # read pkl files which define nodes of interest for TE measurements
        self.content_actors = self.readPickleFile(contentActorsFile)
        self.content_groups = self.readPickleFile(contentFile)

        # set TE parameters
        with open(te_config, 'rb') as f:
            te_params = json.load(f)

        self.startTime = pd.Timestamp(te_params['startTime'])
        self.binSize = te_params['binSize']
        self.teThresh = te_params['teThresh']
        self.delayUnits = np.array(te_params['delayUnits'])
        self.starEvent = te_params['starEvent']
        self.otherEvents = te_params['otherEvents']
        self.kE = te_params['kE']
        self.kN = te_params['kN']
        self.nReps = te_params['nReps']
        self.bGetTS = te_params['bGetTS']

    def preprocess(self, df):

        """
        Edit columns, convert date, sort by date
        """

        if self.platform=='reddit':
            mapping = {'actionType' : 'event',
                       'communityID': 'subreddit',
                       'keywords'   : 'keywords',
                       'nodeID'     : 'content',
                       'nodeTime'   : 'time',
                       'nodeUserID' : 'user',
                       'parentID'   : 'parent',
                       'rootID'     : 'root'}
        elif self.platform=='twitter':
            mapping = {'actionType' : 'event',
                       'nodeID'     : 'content',
                       'nodeTime'   : 'time',
                       'nodeUserID' : 'user',
                       'parentID'   : 'parent',
                       'rootID'     : 'root'}
        elif self.platform=='github':
            mapping = {'nodeID'     : 'content',
                       'nodeUserID' : 'user',
                       'actionType' : 'event',
                       'nodeTime'   : 'time',
                       'actionSubType': 'action',
                       'status':'merged'}
        else:
            print('Invalid platform.')

        df = df.rename(index=str, columns=mapping)

        df = df[df.event.isin(self.popularity_events + self.contribution_events)]

        try:
            df['time'] = pd.to_datetime(df['time'],unit='s')
        except:
            try:
                df['time'] = pd.to_datetime(df['time'],unit='ms')
            except:
                df['time'] = pd.to_datetime(df['time'])


        df = df.sort_values(by='time')
        df = df.assign(time=df.time.dt.floor('h'))
        return df

    def preprocessContentMeta(self, df):
        try:
            df.columns = ['content', 'created_at', 'owner_id', 'language']
        except:
            df.columns = ['created_at', 'owner_id', 'content']
        df['created_at'] = pd.to_datetime(df['created_at'])
        df = df[df.content.isin(self.main_df.content.values)]
        return df

    def preprocessUserMeta(self, df):
        try:
            df.columns = ['user', 'created_at', 'location', 'company']
        except:
            df.columns = ['user', 'created_at', 'city', 'country', 'company']
        df['created_at'] = pd.to_datetime(df['created_at'])
        df = df[df.user.isin(self.main_df.user.values)]
        return df

    def readPickleFile(self, ipFile):

        with open(ipFile, 'rb') as handle:
            obj = pkl.load(handle)

        return obj
