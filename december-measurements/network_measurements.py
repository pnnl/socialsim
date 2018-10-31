from __future__ import print_function, division

import pandas as pd
import igraph as ig
import snap as sn
from time import time
import numpy as np

import community
import tqdm

from prettytable import PrettyTable
from prettytable import MSWORD_FRIENDLY

import os

__all__ = ['GithubNetworkMeasurements',
           'TwitterNetworkMeasurements',
           'RedditNetworkMeasurements']

class NetworkMeasurements(object):
    """
    This class implements Network specific   measurements. It uses iGraph and SNAP libraries with Python interfaces.
    For installation information please visit the websites for the two packages.
 
    iGraph-Python at http://igraph.org/python/
    SNAP Python at https://snap.stanford.edu/snappy/
    """
    def __init__(self, data, test=False):
        self.main_df = data if isinstance(data, pd.DataFrame) else pd.read_csv(data)

        if test:
            print('Running test version of network measurements')
            self.main_df = self.main_df.head(1000)

        assert self.main_df is not None and len(self.main_df) > 0, 'Problem with the dataframe creation'

        self.preprocess()
        
        self.build_undirected_graph(self.main_df)

 
    def preprocess(self):
        return NotImplementedError()

    def build_undirected_graph(self, df):
        return NotImplementedError()
    
    def mean_shortest_path_length(self):
        return sn.GetBfsEffDiamAll(self.gUNsn, 500, False)[3]
    
    def number_of_nodes(self):
        return ig.Graph.vcount(self.gUNig)

    def number_of_edges(self):
        return ig.Graph.ecount(self.gUNig)

    def density(self):
        return ig.Graph.density(self.gUNig)

    def assortativity_coefficient(self):
        return ig.Graph.assortativity_degree(self.gUNig)

    def number_of_connected_components(self):
        return len(ig.Graph.components(self.gUNig,mode="WEAK"))

    def average_clustering_coefficient(self):
        return sn.GetClustCfAll(self.gUNsn, sn.TFltPrV())[0]
        #return ig.Graph.transitivity_avglocal_undirected(self.gUNig,mode="zero")

    def max_node_degree(self):
        return max(ig.Graph.degree(self.gUNig))

    def mean_node_degree(self):
        return 2.0*ig.Graph.ecount(self.gUNig)/ig.Graph.vcount(self.gUNig)

    def degree_distribution(self):
        degVals = ig.Graph.degree(self.gUNig)
        return pd.DataFrame([{'node': idx, 'value': degVals[idx]} for idx in range(self.gUNig.vcount())])

    def community_modularity(self):
        return ig.Graph.modularity(self.gUNig,ig.Graph.community_multilevel(self.gUNig))


    def get_parent_uids(self,df, parent_node_col="parentID", node_col="nodeID", root_node_col="rootID", user_col="nodeUserID"):
        """
        :return: adds parentUserID column with user id of the parent if it exits in df
        if it doesn't exist, uses the user id of the root instead
        if both doesn't exist: NaN
        """
        tweet_uids = pd.Series(df[user_col].values, index=df[node_col]).to_dict()
        df['parentUserID'] = df[parent_node_col].map(tweet_uids)
        df.loc[(df[root_node_col] != df[node_col]) & (df['parentUserID'].isnull()), 'parentUserID'] = \
            df[(df[root_node_col] != df[node_col]) & (df['parentUserID'].isnull())][root_node_col].map(tweet_uids)
        return df

class GithubNetworkMeasurements(NetworkMeasurements):

    def __init__(self, project_on='nodeID', weighted=False, **kwargs):
        self.project_on = project_on
        self.weighted = weighted
        super(GithubNetworkMeasurements, self).__init__(**kwargs)

    def preprocess(self):
        pass

    def build_undirected_graph(self, df):
        
        #self.main_df = pd.read_csv(data)
        self.main_df = self.main_df[['nodeUserID','nodeID']]
        
        left_nodes = np.array(self.main_df['nodeUserID'].unique().tolist())
        right_nodes = np.array(self.main_df['nodeID'].unique().tolist())
        el = self.main_df.apply(tuple, axis=1).tolist()
        edgelist = list(set(el))
 
        #iGraph Graph object construction
        B = ig.Graph.TupleList(edgelist, directed=False)
        names = np.array(B.vs["name"])
        types = np.isin(names,right_nodes)
        B.vs["type"] = types#[name in right_nodes for name in B.vs["name"]]
        p1,p2 = B.bipartite_projection(multiplicity=False) 
        
        self.gUNig = None
        if (self.project_on == "user"):
            self.gUNig = p1
        else:
            self.gUNig = p2
        
        #SNAP graph object construction
        self.gUNsn = sn.TUNGraph.New()
        for v in self.gUNig.vs:
            self.gUNsn.AddNode(v.index)
        for e in self.gUNig.es:
            self.gUNsn.AddEdge(e.source,e.target) 
        

class TwitterNetworkMeasurements(NetworkMeasurements):
    def __init__(self, **kwargs):
        super(TwitterNetworkMeasurements, self).__init__(**kwargs)

    def preprocess(self):
        pass

    def build_undirected_graph(self, df):

        df = self.get_parent_uids(df).dropna(subset=['parentUserID'])
        edgelist = df[['nodeUserID','parentUserID']].apply(tuple,axis=1).tolist()        
                        
        #iGraph Graph object construction
        self.gUNig = ig.Graph.TupleList(edgelist, directed=False)
                
        #SNAP graph object construction
        self.gUNsn = sn.TUNGraph.New()
        for v in self.gUNig.vs:
            self.gUNsn.AddNode(v.index)
        for e in self.gUNig.es:
            self.gUNsn.AddEdge(e.source,e.target)         
        
      
class RedditNetworkMeasurements(NetworkMeasurements):
    def __init__(self, **kwargs):
        super(RedditNetworkMeasurements, self).__init__(**kwargs)

    def preprocess(self):
        pass

    def build_undirected_graph(self,df):

        df = self.get_parent_uids(df).dropna(subset=['parentUserID'])
        edgelist = df[['nodeUserID','parentUserID']].apply(tuple,axis=1).tolist()
                
        #iGraph Graph object construction
        self.gUNig = ig.Graph.TupleList(edgelist, directed=False)
                
        #SNAP graph object construction
        self.gUNsn = sn.TUNGraph.New()
        for v in self.gUNig.vs:
            self.gUNsn.AddNode(v.index)
        for e in self.gUNig.es:
            self.gUNsn.AddEdge(e.source,e.target)        

   
