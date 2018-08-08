'''
The following class is used for the Transfer Entropy Measurements. 

There are three main functions, the rest are helpers. 

The three functions to call are:

computeTEUsers
computeTEUserEvents
computeTERepos

Note computeTEUsersEvents requires TEUsers to have been ran. If computeTEUserEvents is called then it will calculate the computeTEUsers automatically. 

'''
import numpy as np
import pandas as pd
from collections import defaultdict
import jpype
import pickle as pkl
class TEMeasurements():
    def __init__(object):
        super(TE, self).__init__()
        
    '''
    Used For ALL 3 methods
    '''

    def readPickleFile(self,ipFile):

        with open(ipFile, 'rb') as handle:
            obj = pkl.load(handle)

        return obj    
      
    def computeBasicStats(self,timeseries):

        maxTime = 0.0
        for repo,actorsTS in timeseries.iteritems():
            maxTime = max(maxTime,max({key: max(value) for key, value in timeseries[repo].items()}.values()))

        return maxTime 
    
    def getTESigPairsRepo(self,actorTSSrc,actorTSDest,teThresh, delayUnits, nReps, kE, kN):

        actorsSrc = actorTSSrc.keys()
        actorsDest = actorTSDest.keys()

        tSSrc = actorTSSrc.values()
        tSDest = actorTSDest.values()

        nActSrc = len(actorsSrc)
        nActDest = len(actorsDest)

        print("Number of source / destination actors (repos) in this repo (repo group ) : ", nActSrc, " ", nActDest)

        allEdges = {}
        allNodes = {}

        for idxS in range(nActSrc):

            src = tSSrc[idxS]
            nodeTEVal = 0.0

            for idxD in range(nActDest):

                if (actorsSrc[idxS] != actorsDest[idxD]):

                    dest = tSDest[idxD]
                    TEDelays = np.zeros((len(delayUnits)))

                    for idx in range(len(delayUnits)):
                        TEDelays[idx] = self.getTETimeSeriesPairBinary(src, dest, teThresh, delayUnits[idx], nReps)

                    if (np.max(TEDelays) > 0.0):
                        allEdges[tuple((actorsSrc[idxS],actorsDest[idxD]))] = np.max(TEDelays)
                        nodeTEVal = nodeTEVal + np.max(TEDelays)

            if (nodeTEVal > 0.0):
                allNodes[actorsSrc[idxS]] = nodeTEVal

        topEdges = sorted(allEdges.items(), key=lambda (k,v): v, reverse = True)
        if (len(topEdges) > kE):
            topEdges = topEdges[:kE]

        topNodes = sorted(allNodes.items(), key=lambda (k,v): v, reverse = True)
        if (len(topNodes) > kN):
            topNodes = topNodes[:kN]
        
        return (topEdges, topNodes)

    def getTETimeSeriesPairBinary(self,src, dest, teThresh, delayParam, nReps):

        teCalcClass = jpype.JPackage("infodynamics.measures.discrete").TransferEntropyCalculatorDiscrete
        teCalc = teCalcClass(2,1,1,1,1,delayParam)
        teCalc.initialise()
        teCalc.addObservations(src,dest)
        te = teCalc.computeAverageLocalOfObservations()

        if(te > teThresh):
            teNullDist = teCalc.computeSignificance(nReps);
            teNullMean = teNullDist.getMeanOfDistribution()
            teNullStd = teNullDist.getStdOfDistribution()
            if teNullStd > 0:
                z_score = (te-teNullMean)/teNullStd
            else:
                z_score = 0.0
                te = 0.0

            if (z_score < 3.0):
                te = 0.0
        else:
            te = 0.0   

        return te
    
    '''
    For TE Users
    '''
    def getTimeSeriesUsers(self):
        
        df = self.main_df[self.main_df['repo'].isin(self.repo_actors.keys())]
        timeseries = dict()
        for repo in self.repo_actors.keys():
            tempdf = df[df['repo'] == repo]
            if (not tempdf.empty):
                tempdf = df[df['user'].isin(self.repo_actors[repo])]
                if (not tempdf.empty):
                    tempdf['time'] = pd.to_datetime(tempdf['time'])
                    tempdf['time'] = (tempdf['time'] - self.startTime).astype('timedelta64[s]')
                    tempDic = tempdf[['user','time']].groupby('user')['time'].apply(list).to_dict()

                    timeseries[repo] = tempDic

        return timeseries



    def getBinnedBinaryTimeSeries(self,groupEntityTS,binSize,totalBins):

        binnedTS = defaultdict(dict)

        for group,entityTS in groupEntityTS.iteritems():
            entitiesBinnedTS = {}
            for entity, timeSeries in entityTS.iteritems(): 
                entitiesBinnedTS[entity] = self.getBinnedTimeSeriesBinarySingle(totalBins, binSize, timeSeries)

            binnedTS[group] = entitiesBinnedTS

        return binnedTS

    def getBinnedTimeSeriesBinarySingle(self,totalBins,binSize,timeSeries):

        tsBinned = np.zeros((totalBins), dtype=int)
        for timeVal in timeSeries:
            try:
                idx = (timeVal // binSize)
                tsBinned[int(idx)] = 1    
            except:
                continue

        return tsBinned
    

    def createAllTEMatrices(self,rATSrc, rATDest, teThresh, delayUnits, nReps, kE, kN):

        if (set(rATSrc.keys()) != set(rATDest.keys())):
            sys.exit("The repos in the source and target time series data structure is different. Please check.")

        topEdges = defaultdict(dict)
        topNodes = {}

        for repo in rATSrc.keys():

            print("Computing for repo (repo group) : ", repo)
            edges,nodes  = self.getTESigPairsRepo(rATSrc[repo],rATDest[repo],teThresh,delayUnits, nReps, kE, kN) 

            topEdges[repo] = edges
            topNodes[repo] = nodes            

        return (topEdges, topNodes)



    #main function to call
    def computeTEUsers(self):
        #if not jpype.isJVMStarted():
        #    jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + "../infodynamics.jar")
        repoActorsTS = self.getTimeSeriesUsers()

        maxTime = self.computeBasicStats(repoActorsTS)
        totalBins = int(np.ceil(maxTime/float(self.binSize)))

        repoActorsBinned = self.getBinnedBinaryTimeSeries(repoActorsTS, self.binSize, totalBins)

        topEdges, topNodes = self.createAllTEMatrices(repoActorsBinned, repoActorsBinned, self.teThresh[0], self.delayUnits, nReps = self.nReps, kE=self.kE, kN=self.kN) 
        #jpype.shutdownJVM()
        self.top_edges = topEdges
        self.top_users = topNodes
        #with open('top_users.pkl','w') as handle:
        #    pkl.dump(allDic,handle)
        return topEdges, topNodes  
    
    
  
    
    '''
    Compute TEUSerEvents
    '''
    def createAllTEMatrices(self, rATSrc, rATDest, teThresh, delayUnits, nReps, kE, kN):
    
        if (set(rATSrc.keys()) != set(rATDest.keys())):
            sys.exit("The repos in the source and target time series data structure is different. Please check.")

        topEdges = defaultdict(dict)
        topNodes = {}

        for repo in rATSrc.keys():

            print("Computing for repo (repo group) : ", repo)
            edges,nodes  = self.getTESigPairsRepo(rATSrc[repo],rATDest[repo],teThresh,delayUnits, nReps, kE, kN) 
           
            topEdges[repo] = edges
            topNodes[repo] = nodes            

        return (topEdges, topNodes)
    
    def getSourceTargetUserEventTS(self,repoActorEventsTS,repoRockstars, rockStarEvent, otherEvents):
    
        repoActorsSRC = defaultdict(dict)
        repoActorsTAR  = defaultdict(dict)
        reposConsidered = repoRockstars.keys()

        #First the rockstars who will act as sources
        for repo,actorEventTS in repoActorEventsTS.iteritems():
            if (repo in reposConsidered):
                rockStars = [x[0] for x in repoRockstars[repo]]
                for actor,eventTS in actorEventTS.iteritems():
                    if ((actor in rockStars) and (rockStarEvent in eventTS.keys())):
                        if (len(eventTS[rockStarEvent]) > 20):
                            repoActorsSRC[repo][actor] = eventTS[rockStarEvent]

        #The other users who form the targets
        for repo,actorEventTS in repoActorEventsTS.iteritems():
            if (repo in reposConsidered):
                rockStars = [x[0] for x in repoRockstars[repo]]
                for actor,eventTS in actorEventTS.iteritems():
                    if (actor not in rockStars):
                        combinedEvent = []
                        for event in otherEvents:
                            if (event in eventTS.keys()):
                                combinedEvent = combinedEvent + eventTS[event]
                        if (len(combinedEvent) > 20):
                            repoActorsTAR[repo][actor] = combinedEvent


        #Ensure that both SRC and TAR contain exactly the same repos since filtering criteria are different
        srcKeys = repoActorsSRC.keys()
        tarKeys = repoActorsTAR.keys()
        differenceKeys = []
        if (len(srcKeys) > len(tarKeys)):
            differenceKeys = list(set(srcKeys).difference(set(tarKeys)))
            for diffkey in differenceKeys:
                del repoActorsSRC[diffkey]       
        elif (len(tarKeys) > len(srcKeys)):
            differenceKeys = list(set(tarKeys).difference(set(srcKeys)))
            for diffkey in differenceKeys:
                    del repoActorsTAR[diffkey]                          


        return (repoActorsSRC, repoActorsTAR)
    
    
    def getTimeSeriesUsersEvents(self,df,repoActors):
        
        df = df[df['repo'].isin(repoActors.keys())]

        timeseries = dict()
        for repo in repoActors.keys():
            tempdf = df[df['repo'] == repo]

            if len(tempdf) == 0:
                timeseries[repo] = dict()
                continue
            tempdf = df[df['user'].isin(repoActors[repo])]

            if len(tempdf) == 0:
                timeseries[repo] = dict()
                continue

            tempdf['time'] = pd.to_datetime(tempdf['time'])
            tempdf['time'] = (tempdf['time'] - self.startTime).astype('timedelta64[s]')
        
            tempdf = pd.DataFrame(tempdf[['user','event','time']].groupby(['user','event'])['time'].apply(list))

            tempdf = tempdf.reset_index()
            tempdic = dict()

            for ele in tempdf['user'].unique():
                tm = dict()
                curdf = tempdf[tempdf['user'] == ele]
                for eventT in curdf['event'].unique():
                    tm[eventT] = curdf[curdf['event'] == eventT]['time'].values[0]
                tempdic[ele] = tm
            timeseries[repo] = tempdic
        return timeseries
    
    def computeTEUserEvents(self):
        repoActorEventsTS = self.getTimeSeriesUsersEvents(self.main_df, self.repo_actors) 

        if len(self.top_users) == 0:
            self.computeTEUsers()

        repoRockstars = self.top_users
#         #Divide up the data into SRC (rockstars) and TAR (others) time series
        repoActorsSRC, repoActorsTAR = self.getSourceTargetUserEventTS(repoActorEventsTS,repoRockstars, self.starEvent, self.otherEvents)

        #Get binned time series
        maxT = max(self.computeBasicStats(repoActorsSRC), self.computeBasicStats(repoActorsTAR))
        totalBins = int(np.ceil(maxT/float(self.binSize)))

        repoActorsSRCBinned = self.getBinnedBinaryTimeSeries(repoActorsSRC, self.binSize, totalBins)
        repoActorsTARBinned = self.getBinnedBinaryTimeSeries(repoActorsTAR, self.binSize, totalBins)

        topEdges, topNodes = self.createAllTEMatrices(repoActorsSRCBinned, repoActorsTARBinned, self.teThresh[1], self.delayUnits, nReps = self.nReps, kE=self.kE, kN=self.kN) 
        return topEdges, topNodes
        
    
    '''
    TE REPOS
       
    '''
    def getTimeSeriesRepos(self):
  
        timeseries = dict()
        for desc,repos in self.repo_groups.iteritems():
            tempdf = self.main_df[self.main_df['repo'].isin(repos)] #get only repos we care about
            if (not tempdf.empty):
                tempdf['time'] = pd.to_datetime(tempdf['time'])
                tempdf['time'] = (tempdf['time'] - self.startTime).astype('timedelta64[s]')
                tempDic = tempdf[['repo','time']].groupby('repo')['time'].apply(list).to_dict()
                timeseries[desc] = tempDic

        return timeseries    
    
    def computeTERepos(self):
        print("Getting time series from CSV data file.")
        repoTS = self.getTimeSeriesRepos()

        #Get binned time series
        maxT = self.computeBasicStats(repoTS)
        totalBins = int(np.ceil(maxT/float(self.binSize)))
        reposBinned = self.getBinnedBinaryTimeSeries(repoTS, self.binSize, totalBins)    

        topEdges, topNodes = self.createAllTEMatrices(reposBinned, reposBinned, self.teThresh[2], self.delayUnits, nReps = self.nReps, kE = self.kE, kN = self.kN) 
        return topEdges, topNodes
