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
import json

class Measurements(UserCentricMeasurements, RepoCentricMeasurements, TEMeasurements, CommunityCentricMeasurements):
    def __init__(self, dfLoc, interested_repos=[], interested_users=[], metaRepoData=False, metaUserData=False,
                 repoActorsFile='data/filtUsers-test.pkl',reposFile='data/filtRepos-test.pkl',topNodes=[],topEdges=[],
                 previousActionsFile='',community_dictionary='data/communities.pkl',te_config='te_params_dry_run2.json'):
        super(Measurements, self).__init__()
        
        try:
            #check if input is a data frame
            dfLoc.columns
            df = dfLoc
        except:
            #if not it should be a csv file path
            df = pd.read_csv(dfLoc)

        self.contribution_events = ["PullRequestEvent", "PushEvent", "IssuesEvent","IssueCommentEvent","PullRequestReviewCommentEvent","CommitCommentEvent","CreateEvent"]
        self.popularity_events = ['WatchEvent','ForkEvent']

        print('preprocessing...')
        self.main_df = self.preprocess(df)

        print('splitting optional columns...')
        #store action and merged columns in a seperate data frame that is not used for most measurements
        if len(self.main_df.columns) == 6:
            self.main_df_opt = self.main_df.copy()[['action','merged']]
            self.main_df_opt['merged'] = self.main_df_opt['merged'].astype(bool)
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
        self.communities = self.getCommunities(path=community_dictionary)

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
        with open(te_config,'rb') as f:
            te_params = json.load(f)

        self.startTime = pd.Timestamp(te_params['startTime'])
        self.binSize= te_params['binSize']
        self.teThresh = te_params['teThresh']
        self.delayUnits = np.array(te_params['delayUnits'])
        self.starEvent = te_params['starEvent']
        self.otherEvents = te_params['otherEvents']
        self.kE = te_params['kE']
        self.kN = te_params['kN']
        self.nReps = te_params['nReps']
        self.bGetTS = te_params['bGetTS']



    def preprocess(self,df):
        #edit columns, convert date, sort by date
        if df.columns[0] == '_id':
            del df['_id']
        if len(df.columns) == 4:
            df.columns = ['time', 'event', 'user', 'repo']
        else:
            df.columns = ['time', 'event', 'user', 'repo','action','merged']
        df = df[df.event.isin(self.popularity_events + self.contribution_events)]
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values(by='time')
        df = df.assign(time=df.time.dt.floor('h'))
        return df

    def preprocessRepoMeta(self,df):
        try:
            df.columns = ['repo','created_at','owner_id','language']
        except:
            df.columns = ['created_at','owner_id','repo']
        df = df[df.repo.isin(self.main_df.repo.values)]
        df['created_at'] = pd.to_datetime(df['created_at'])
        #df = df.drop_duplicates('repo')
        return df
    
    def preprocessUserMeta(self,df):
        try:
            df.columns = ['user','created_at','location','company']
        except:
            df.columns = ['user','created_at','city','country','company']
        
        df = df[df.user.isin(self.main_df.user.values)]
        df['created_at'] = pd.to_datetime(df['created_at'])
        return df

    def readPickleFile(self,ipFile):

        with open(ipFile, 'rb') as handle:
            obj = pkl.load(handle)

        return obj 
