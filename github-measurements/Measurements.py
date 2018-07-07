import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import Pool
from functools import partial
from pathos import pools as pp
import pickle as pkl
from UserCentricMeasurements import *
from RepoCentricMeasurements import *
from CommunityCentricMeasurements import *
from TEMeasurements import *
from collections import defaultdict
import jpype

class Measurements(UserCentricMeasurements, RepoCentricMeasurements, TEMeasurements, CommunityCentricMeasurements):
    def __init__(self, dfLoc, interested_repos=[], interested_users=[], metaRepoData=False, metaUserData=False,
                 repoActorsFile='data/filtUsers-test.pkl',reposFile='data/filtRepos-test.pkl',topNodes=[],topEdges=[],
                 previousActionsFile='data/prior_contribution_counts.csv'):
        super(Measurements, self).__init__()
        
        try:
            #check if input is a data frame
            dfLoc.columns
            df = dfLoc
        except:
            #if not it should be a csv file path
            df = pd.read_csv(dfLoc)

        self.contribution_events = ["PullRequestEvent", "PushEvent", "IssuesEvent","IssueCommentEvent","PullRequestReviewCommentEvent","CommitCommentEvent","CreateEvent"]

        print('preprocessing...')
        self.main_df = self.preprocess(df)

        print('splitting optional columns...')
        #store action and merged columns in a seperate data frame that is not used for most measurements
        if len(self.main_df.columns) == 6:
            self.main_df_opt = self.main_df.copy()[['action','merged']]
            self.main_df = self.main_df.drop(['action','merged'],axis=1)
        else:
            self.main_df_opt = None


        #For repoCentric
        print('getting selected repos...')
        self.selectedRepos = self.getSelectRepos(interested_repos) #Dictionary of selected repos index == repoid

        #For userCentric
        self.selectedUsers = self.main_df[self.main_df.user.isin(interested_users)]



        print('processing repo metatdata...')
        #read in external metadata files
        #repoMetaData format - full_name_h,created_at,owner.login_h,language
        #userMetaData format - login_h,created_at,location,company
        if metaRepoData != False:
            self.useRepoMetaData = True
            self.repoMetaData = self.preprocessRepoMeta(pd.read_csv(metaRepoData))
        else:
            self.useRepoMetaData = False
        print('processing user metatdata...')
        if metaUserData != False:
            self.useUserMetaData = True
            self.userMetaData = self.preprocessUserMeta(pd.read_csv(metaUserData))
        else:
            self.useUserMetaData = False


        #For Community
        print('getting communities...')
        self.communities = self.getCommunities()

        #read in previous events count external file (used only for one measurement)
        try:
            print('reading previous counts...')
            self.previous_event_counts = pd.read_csv(previousActionsFile)
        except:
            self.previous_event_counts = None


        #For TE
        print('starting jvm...')
        if not jpype.isJVMStarted():
            jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + "infodynamics.jar")

        self.top_users = topNodes
        self.top_edges = topEdges

        #read pkl files which define nodes of interest for TE measurements
        self.repo_actors = self.readPickleFile(repoActorsFile)
        self.repo_groups = self.readPickleFile(reposFile)

        #set TE parameters
        self.startTime = pd.Timestamp('2017-07-01 00:00:00')
        self.binSize = 3600
        self.teThresh = [0.01,0.0075,0.0075]
        self.delayUnits = np.linspace(3,24,8).astype(int)
        self.starEvent = 'IssueCommentEvent'
        self.otherEvents = ['PushEvent','IssuesEvent','IssueCommentEvent','PullRequestEvent']
        self.kE = 50
        self.kN = 12
        self.nReps = 100
        self.bGetTS = True

    def preprocess(self,df):
        #edit columns, convert date, sort by date
        if df.columns[0] == '_id':
            del df['_id']
        if len(df.columns) == 4:
            df.columns = ['time', 'event', 'user', 'repo']
        else:
            df.columns = ['time', 'event', 'user', 'repo','action','merged']
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values(by='time')
        return df

    def preprocessRepoMeta(self,df):
        df.columns = ['repo','created_at','owner_id','language']
        df['created_at'] = pd.to_datetime(df['created_at'])
        return df

    def preprocessUserMeta(self,df):
        df.columns = ['user','created_at','location','company']
        df['created_at'] = pd.to_datetime(df['created_at'])
        return df

    def readPickleFile(self,ipFile):

        with open(ipFile, 'rb') as handle:
            obj = pkl.load(handle)

        return obj 
