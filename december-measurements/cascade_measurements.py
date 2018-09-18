import networkx as nx
import pandas as pd

from cascade_validators import check_root_only
from validators import check_empty
import pysal
import numpy as np
from collections import Counter


"""
IMPORTANT: all timestamp fields in the dfs must be in pandas datetime format
"""

def palma_ratio(income):
    income = np.sort(np.array(income))
    percent_nodes = np.arange(1, len(income) + 1) / float(len(income))
    # percent of events taken by top 10% of nodes
    p10 = np.sum(income[percent_nodes >= 0.9])
    # percent of events taken by bottom 40% of nodes
    p40 = np.sum(income[percent_nodes <= 0.4])
    try:
        p = float(p10) / float(p40)
    except ZeroDivisionError:
        return None
    return p


def get_original_tweet_ratio(main_df, node_col, root_node_col):
    original_tweets_count = len(main_df[main_df[node_col] == main_df[root_node_col]])
    replies_count = len(main_df[main_df["actionType"] == "reply"])
    retweets_count = len(main_df[main_df["actionType"] == "retweet"])
    return (original_tweets_count + replies_count) / (original_tweets_count + replies_count + retweets_count)



class Cascade:
    """
        depth, max breadth, size and structural virality measurements from
        Soroush Vosoughi, Deb Roy, and Sinan Aral.
        The spread of true and false news online. Science. 2018
    """

    def __init__(self, main_df=None, parent_node_col="parentID", node_col="nodeID", root_node_col="rootID", timestamp_col="nodeTime", user_col = "nodeUserID"):
        """
        main_df: df containing all tweets in the RT cascade of the original tweet
        parent_node: name of the column containg the uid of the node who was retweeted from (None for cascade root)
        node: name of the column containg the uid of the node who retweeted from node in parent_node
        timestamp: time of the original tweet/retweet
        """
        self.parent_node_col = parent_node_col
        self.node_col = node_col
        self.root_node_col = root_node_col
        self.timestamp_col = timestamp_col
        self.user_col = user_col
        if main_df is not None:
            if len(main_df) > 0:
                self.preprocess_and_create_nx(main_df, set_cascade=True)
            else:
                self.main_df = main_df

    def preprocess_and_create_nx(self, main_df, set_cascade=True):
        root_df = main_df[main_df[self.node_col] == main_df[self.root_node_col]]
   
        if len(root_df) > 0:
            self.root_node = root_df[self.node_col].values[0]
        else:
            self.root_node = main_df[self.root_node_col].values[0]

        if set_cascade:
            self.main_df = main_df
            self.cascade_nx = nx.from_pandas_edgelist(main_df,
                                                      target=self.parent_node_col, source=self.node_col,
                                                      create_using=nx.DiGraph())
                      
            #self.cascade_nx = nx.from_pandas_edgelist(main_df[main_df[self.node_col] != main_df[self.root_node_col]],
            #                                          target=self.parent_node_col, source=self.node_col,
            #                                          create_using=nx.DiGraph())
        else:
            return main_df
        
    def update_cascade(self, df):
        if not hasattr(self, 'main_df'):
            self.main_df = df
            self.cascade_nx = nx.from_pandas_edgelist(self.main_df[self.main_df[self.node_col] != self.main_df[self.root_node_col]],
                                                      target=self.parent_node_col,
                                                      source=self.node_col, create_using=nx.DiGraph())
        else:
            self.main_df = pd.concat([self.main_df, df])
            self.cascade_nx.add_edges_from([tuple(x) for x in df[[self.node_col, self.parent_node_col]].values])


    @check_empty(default=None)
    #@check_root_only(default=0)
    def get_cascade_depth(self):
 
        path_lengths = []
        for x in self.cascade_nx.nodes():
            if self.cascade_nx.in_degree(x) == 0:
                try:
                    p_len = nx.shortest_path_length(self.cascade_nx, x, self.root_node)
                    path_lengths.append(p_len)
                except:
                    ''
        if len(path_lengths) > 0:
            return max(path_lengths)
        else:
            return None

    @check_empty(default=0)
    #@check_root_only(default=1)
    def get_cascade_size(self):
        return nx.number_of_nodes(self.cascade_nx)

    @check_empty(default=None)
    #@check_root_only(default=0)
    def get_cascade_breadth(self):
        degrees = [self.cascade_nx.in_degree(x) for x in self.cascade_nx.nodes()]
        if len(degrees) > 0:
            return max(degrees)
        else:
            return None

    @check_empty(default=None)
    #@check_root_only(default=None)
    def get_cascade_structural_virality(self):
        """
        :return: structural virality of a single cascade.
                 For definition:
                 Soroush Vosoughi, Deb Roy, Sinan Aral. The spread of true and false news online. Science. 2018
        """
        try:
            cascade_nx_undirected = max(nx.connected_component_subgraphs(self.cascade_nx.to_undirected()),key=len)
        except:
            cascade_nx_undirected = self.cascade_nx.to_undirected()
        n = nx.number_of_nodes(cascade_nx_undirected)
        try:
            return nx.wiener_index(cascade_nx_undirected) * 2 / (n * (n - 1))
        except:
            return None

    @check_empty(default=0)
    ##@check_root_only(default=1)
    def get_cascade_nodes(self, unique=True):
        if unique:
            return set(self.main_df[self.user_col])
        else:
            return self.main_df[self.node_col]

    @check_empty(default=None)
    #@check_root_only(default=0)
    def get_cascade_lifetime(self, granularity="D"):
        """
        :param granularity: "s", "m", "H", "D"  [seconds/minutes/days/hours]
        """
        try:
            lifetime = (max(self.main_df[self.timestamp_col]) - min(self.main_df[self.timestamp_col])).total_seconds()
        except:
            lifetime = (max(self.main_df[self.timestamp_col]) - min(self.main_df[self.timestamp_col]))
        if granularity in ["m", "H", "D"]:
            lifetime /= 60
        if granularity in ["H", "D"]:
            lifetime /= 60
        if granularity == "D":
            lifetime /= 24
        return lifetime

    def get_cascade_original_tweet_ratio(self):
        """
        Twitter only measurement
        """
        get_original_tweet_ratio(self.main_df, self.node_col, self.root_node_col)


class SingleCascadeMeasurements:
    """
    measurements for a cascade i.e. node level measurements
    """

    def __init__(self, main_df, parent_node_col="parentID", node_col="nodeID", root_node_col="rootID", timestamp_col="nodeTime", user_col = "nodeUserID"):
        self.parent_node_col = parent_node_col
        self.node_col = node_col
        self.root_node_col = root_node_col
        self.timestamp_col = timestamp_col
        self.user_col = user_col
        self.cascade = Cascade(parent_node_col=self.parent_node_col, node_col=self.node_col, root_node_col=root_node_col, timestamp_col=self.timestamp_col,user_col=self.user_col)
        if main_df is not None:
            if len(main_df) > 0:
                self.main_df = self.cascade.preprocess_and_create_nx(main_df, set_cascade=False)
            else:
                self.main_df = main_df

        self.temporal_measurements = {}
        self.depth_based_measurement_df = None

        try:
            self.main_df[timestamp_col] = pd.to_datetime(self.main_df[timestamp_col],unit='s')
        except:
            self.main_df[timestamp_col] = pd.to_datetime(self.main_df[timestamp_col],unit='ms')


    @check_empty(default=None)
    ##@check_root_only(default=None)
    def get_temporal_measurements(self, time_granularity="M"):
        """
        :param time_granularity: "Y", "M", "D", "H" [years/months/days/hours]
        :return: pandas dataframe with "breadth", "size", "structural_virality", "unique_nodes", "new_node_ratio" at each timestamp
        """
        temporal_measurements = []
        old_unique_nodes_count = 1  # root node, since we start iterating from depth 1
        for ts, df in self.main_df.set_index(self.timestamp_col).groupby(pd.Grouper(freq=time_granularity), sort=True):
            self.cascade.update_cascade(df)
            if len(df) == 1: # root only
                continue
            old_unique_nodes_count, temporal_measurement = self.get_incremental_cascade_measurements(ts, old_unique_nodes_count)
            temporal_measurements.append(temporal_measurement)
        self.temporal_measurements[time_granularity] = pd.DataFrame(temporal_measurements,
                            columns=["timestamp", "depth", "breadth", "size", "structural_virality", "unique_nodes", "new_node_ratio"])

    def cascade_timeseries_of(self, attribute, time_granularity):
        """
        :param attribute: "depth", "breadth", "size", "structural_virality", "unique_nodes", "new_node_ratio"
        :param time_granularity: "Y", "M", "D", "H" [years/months/days/hours]
        """
        if time_granularity not in self.temporal_measurements:
            self.get_temporal_measurements(time_granularity)
        meas = self.temporal_measurements[time_granularity][["timestamp",attribute]]
        meas.fillna(value=np.nan,inplace=True)
        meas.columns = ['time','value']
        print('meas',meas)
        return meas


    @check_empty(default=None)
    #@check_root_only(default=None)
    def get_depth_based_measurements(self):
        """
        :return: pandas dataframe with "breadth", "size", "structural_virality", "unique_nodes", "new_node_ratio" at each depth
        """
        self.main_df["depth"] = 0
        seed_nodes = [self.cascade.root_node]
        depth = 1
        while len(seed_nodes) > 0:
            self.main_df.loc[self.main_df[self.parent_node_col].isin(seed_nodes), 'depth'] = depth
            seed_nodes = self.main_df[ (self.main_df[self.parent_node_col].isin(seed_nodes)) & (self.main_df[self.node_col] != self.main_df[self.parent_node_col]) ][self.node_col].values
            assert len(set(seed_nodes)) == len(seed_nodes)
            depth += 1
        depth_based_measurements = []
        old_unique_nodes_count = 1  # root node, since we start iterating from depth 1
        self.cascade.update_cascade(self.main_df[self.main_df["depth"] == 0])  # initialize with root
        for depth in range(1, max(self.main_df['depth'])+1):
            self.cascade.update_cascade(self.main_df[self.main_df["depth"] == depth])
            old_unique_nodes_count, depth_based_measurement = self.get_incremental_cascade_measurements(depth, old_unique_nodes_count, by_depth=True)
            depth_based_measurements.append(depth_based_measurement)
        self.depth_based_measurement_df = pd.DataFrame(depth_based_measurements, columns=["depth", "breadth", "size", "structural_virality", "unique_nodes", "new_node_ratio"])

    def cascade_depth_by(self, attribute):
        """
        :param attribute: "breadth", "size", "structural_virality", "unique_nodes", "new_node_ratio"
        """
        if self.depth_based_measurement_df is None:
            self.get_depth_based_measurements()
        meas = self.depth_based_measurement_df[["depth", attribute]]
        meas.columns = ['depth','value']
        return meas


    def get_incremental_cascade_measurements(self, grouper_value, old_unique_nodes_count, by_depth=False):
        unique_nodes_count = len(self.cascade.get_cascade_nodes(unique=True))
        all_measurements = [grouper_value,
                             self.cascade.get_cascade_depth(),
                             self.cascade.get_cascade_breadth(),
                             self.cascade.get_cascade_size(),
                             self.cascade.get_cascade_structural_virality(),
                             unique_nodes_count,
                             (unique_nodes_count - old_unique_nodes_count) / unique_nodes_count]
        if by_depth:
            all_measurements = all_measurements[1:]
        return unique_nodes_count, all_measurements

    @check_empty(default=None)
    #@check_root_only(default=None)
    def cascade_participation_gini(self):
        return pysal.inequality.gini.Gini(self.node_participation()).g

    @check_empty(default=None)
    #@check_root_only(default=None)
    def cascade_participation_palma(self):
        return palma_ratio(self.node_participation())

    def node_participation(self):
        return self.main_df.groupby(self.main_df[self.node_col]).size().reset_index(name='counts')['counts'].values

    def fraction_of_nodes_with_outside_links(self):
        pass


class CascadeCollectionMeasurements:

    def __init__(self, main_df, parent_node_col="parentID", node_col="nodeID", root_node_col="rootID", timestamp_col="nodeTime",user_col = "nodeUserID",
                 filter_on_col=None, filter_in_list=[]):
        """
        main_df: df containing all original tweets/posts and the tweets/comments in their cascades
        parent_node: name of the column containg the uid of the node who was retweeted from
        node: name of the column containg the uid of the node who retweeted from node in parent_node
        timestamp: time of the original tweet/retweet
        """
        self.main_df = main_df
        self.root_node_col = root_node_col
        self.parent_node_col = parent_node_col
        self.node_col = node_col
        self.timestamp_col = timestamp_col
        self.user_col = user_col
        self.filter_on_col = filter_on_col
        self.filter_in_list = filter_in_list
        # for reddit community measurements
        if self.filter_on_col is not None and len(filter_in_list) > 0:
            self.main_df = self.main_df[self.main_df[self.filter_on_col].isin(self.filter_in_list)]
        self.preprocess_and_create_nx_dict()
        self.cascade_distribution_measurement_df = None
        self.community_users_count_timeseries_df = {}

        try:
            self.main_df[timestamp_col] = pd.to_datetime(self.main_df[timestamp_col],unit='s')
        except:
            self.main_df[timestamp_col] = pd.to_datetime(self.main_df[timestamp_col],unit='ms')

        try:
            self.main_df['communityID'] = self.main_df['nodeAttributes'].apply(lambda x: eval(x)['communityID'])
        except:
            ''   

    def preprocess_and_create_nx_dict(self):
        self.cascades = {}
        for cascade_identifier, cascade_df in self.main_df.groupby(self.root_node_col):
            self.cascades[cascade_identifier] = Cascade(main_df=cascade_df, parent_node_col=self.parent_node_col, root_node_col=self.root_node_col, node_col=self.node_col, timestamp_col=self.timestamp_col, user_col=self.user_col)

    @check_empty(default=None)
    #@check_root_only(default=None)
    def get_cascades_distribution_measurements(self):
        """
        :return: pandas dataframe with cascade identiifer and "depth", "breadth", "size", "structural_virality" and lifetime for each cascade in the population
        """
        cascades_distribution_measurements = []
        for cascade_identifier, cascade in self.cascades.items():
            cascades_distribution_measurements.append([cascade_identifier,
                                                       cascade.get_cascade_depth(),
                                                       cascade.get_cascade_size(),
                                                       cascade.get_cascade_breadth(),
                                                       cascade.get_cascade_structural_virality(),
                                                       cascade.get_cascade_lifetime()
                                                       ])
        self.cascade_distribution_measurement_df = pd.DataFrame(cascades_distribution_measurements, columns=["rootID", "depth", "size", "breadth", "structural_virality", "lifetime"])

    def cascade_collection_distribution_of(self, attribute):
        """
        :param attribute: "depth", "size", "breadth", "structural_virality", "lifetime"
        """
        if self.cascade_distribution_measurement_df is None:
            self.get_cascades_distribution_measurements()
        meas = self.cascade_distribution_measurement_df[["rootID",attribute]]
        meas.columns = ['content','value']
        return meas

    @check_empty(default=None)
    #@check_root_only(default=None)
    def get_cascade_collection_timeline_timeseries(self, time_granularity="M", community_grouper=None):
        """
         :param time_granularity: "Y", "M", "D", "H" [years/months/days/hours]
         :param community_grouper: column that indicates a community, eg. communityID, keyword
         :return: pandas dataframe with mean "lifetime" of cascades that start in that interval
         """
        temporal_measurements = []
        result_df_columns = ["timestamp", "value"]
        grouper = [pd.Grouper(freq=time_granularity)]
        if community_grouper and community_grouper in self.main_df.columns:
            grouper.append(community_grouper)
            result_df_columns = ["timestamp", community_grouper, "value"]
        for ts, df in self.main_df[self.main_df[self.node_col] == self.main_df[self.root_node_col]]. \
                set_index(self.timestamp_col).groupby(grouper, sort=True):
            mean_lifetime = df.groupby(self.root_node_col).size().mean()

            temporal_measurements.append(list(ts) + [mean_lifetime] if community_grouper and community_grouper in self.main_df.columns else [ts, mean_lifetime])
            #temporal_measurements.append([*ts, mean_lifetime] if community_grouper else [ts, mean_lifetime])
        return pd.DataFrame(temporal_measurements, columns=result_df_columns)


    @check_empty(default=None)
    #@check_root_only(default=None)
    def get_cascade_collection_size_timeseries(self, time_granularity="M", community_grouper=None):
        """
        :param time_granularity: "Y", "M", "D", "H" [years/months/days/hours]
         :param community_grouper: column that indicates a community, eg. communityID, keyword
        :return: pandas dataframe with mean "size" of cascades that start in that interval
        """
        temporal_measurements = []
        result_df_columns = ["timestamp", "value"]
        grouper = [pd.Grouper(freq=time_granularity)]
        if community_grouper and community_grouper in self.main_df.columns:
            grouper.append(community_grouper)
            result_df_columns = ["timestamp", community_grouper, "value"]
        for ts, df in self.main_df[self.main_df[self.node_col] == self.main_df[self.root_node_col]]. \
                set_index(self.timestamp_col).groupby(grouper, sort=True):
            if len(df.index) > 0:
                mean_size = sum([self.cascades[cascade_identifier].get_cascade_size() for cascade_identifier in df[self.root_node_col].values]) / len(df)
                temporal_measurements.append(list(ts) + [mean_size] if community_grouper and community_grouper in self.main_df.columns else [ts, mean_size])
            #temporal_measurements.append([*ts, mean_size] if community_grouper else [ts, mean_size])
        meas = pd.DataFrame(temporal_measurements, columns=result_df_columns)
        if len(meas.index) == 0:
            return None
        else:
            return meas

    @check_empty(default=None)
    #@check_root_only(default=None)
    def get_community_users_count_timeseries(self, time_granularity="M", community_grouper=None):
        """
        :param time_granularity: "Y", "M", "D", "H" [years/months/days/hours]
         :param community_grouper: column that indicates a community, eg. communityID, keyword
        :return: pandas dataframe with number of unique users who participate in start in that interval
        """
                             # (unique_nodes_count - old_unique_nodes_count) / unique_nodes_count]
        temporal_measurements = []

        if community_grouper in self.main_df.columns:
            for community_identifier, community_df in self.main_df.groupby(community_grouper):
                cumul_df = None
                for ts, df in community_df.set_index(self.timestamp_col).groupby(pd.Grouper(freq=time_granularity), sort=True):
                    if cumul_df is None:
                        cumul_df = df.copy()
                        old_unique_users_count = 0
                    else:
                        old_unique_users_count = cumul_df[self.user_col].nunique()
                        cumul_df.append(df, ignore_index=True)
                    unique_users_count = cumul_df[self.user_col].nunique()
                    new_users_ratio = (unique_users_count - old_unique_users_count) / unique_users_count
                    temporal_measurements.append([ts, community_identifier, unique_users_count, new_users_ratio])
                self.community_users_count_timeseries_df[time_granularity] = pd.DataFrame(temporal_measurements, columns=["timestamp", community_grouper, "unique_users", "new_user_ratio"])

    def community_users_count(self, attribute, time_granularity, community_grouper):
        """
        :param attribute: "unique_users", "new_user_ratio"
        :param time_granularity: "Y", "M", "D", "H" [years/months/days/hours]
        :param community_grouper: column that indicates a community, eg. communityID, keyword
        """
        if community_grouper in self.main_df.columns:
            if time_granularity not in self.community_users_count_timeseries_df:
                self.get_community_users_count_timeseries(time_granularity,community_grouper=community_grouper)
            df = self.community_users_count_timeseries_df[time_granularity][["timestamp", community_grouper, attribute]]
            df.columns = ["timestamp",community_grouper,"value"]
            return df
        else:
            return None

    @check_empty(default=None)
    #@check_root_only(default=None)
    def cascade_collection_initialization_gini(self):
        root_nodes = [cascade.root_node for cascade in self.cascades.values()]
        return pysal.inequality.gini.Gini(list(Counter(root_nodes).values())).g

    @check_empty(default=None)
    #@check_root_only(default=None)
    def cascade_collection_initialization_palma(self):
        root_nodes = [cascade.root_node for cascade in self.cascades.values()]
        return palma_ratio(list(Counter(root_nodes).values()))

    @check_empty(default=None)
    #@check_root_only(default=None)
    def cascade_collection_participation_gini(self):
        all_nodes = [node for cascade in self.cascades.values() for node in cascade.get_cascade_nodes(unique=False)]
        return pysal.inequality.gini.Gini(list(Counter(all_nodes).values())).g

    @check_empty(default=None)
    #@check_root_only(default=None)
    def cascade_collection_participation_palma(self):
        all_nodes = [node for cascade in self.cascades.values() for node in cascade.get_cascade_nodes(unique=False)]
        return palma_ratio(list(Counter(all_nodes).values()))

    @check_empty(default=None)
    #@check_root_only(default=None)
    def fraction_of_nodes_in_lcc(self):
        return max([cascade.get_cascade_size() for cascade in self.cascades.values()]) / len(self.main_df)

    def fraction_of_isolated_nodes(self):
        """not applicable since we do not consider isolated nodes as cascades"""
        pass

    def fraction_of_nodes_with_outside_links(self):
        """ we might not have url information in simulations """
        pass

    def original_tweet_ratio(self):
        """
        Twitter only measurement
        """
        get_original_tweet_ratio(self.main_df, self.node_col, self.root_node_col)


