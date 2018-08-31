from __future__ import print_function, division
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
import community
import tqdm

__all__ = ['GithubNetworkMeasurements', 'TwitterNetworkMeasurements', 'RedditNetworkMeasurements']


class NetworkMeasurements(object):
    """
    This class implements Network specific   measurements.

    """

    def __init__(self, data, node1='user', node2='repo', test=True):
        self.main_df = data if isinstance(data, pd.DataFrame) else pd.read_csv(data)

        if test:
            self.main_df = self.main_df.head(1000)

        assert self.main_df is not None and len(self.main_df) > 0, "Problem with the dataframe creation"

        self.node1 = node1

        self.node2 = node2

        self.preprocess()
        
        self.build_undirected_graph(self.main_df)

        self.all_path_lens = self.node_path_distribution()

    def preprocess(self):
        return NotImplementedError()

    def build_undirected_graph(self, df):
        return NotImplementedError()

    def node_path_distribution(self):
        all_paths = []
        for source_node, target_nodes in nx.shortest_path_length(self.G_undirected):
            # print(source_node, target_nodes)
            for target_node, dist in target_nodes.items():
                if target_node != source_node:
                    all_paths.append({'node1': source_node, 'node2': target_node, 'value': dist})
        return pd.DataFrame(all_paths)

    def number_of_nodes(self):
        return nx.number_of_nodes(self.G_undirected)

    def number_of_edges(self):
        return nx.number_of_edges(self.G_undirected)

    def density(self):
        return nx.density(self.G_undirected)

    def shortest_path_length_distribution(self):
        return self.all_path_lens

    def min_shortest_path_length(self):
        return min(self.all_path_lens['value'].values)

    def max_shortest_path_length(self):
        return max(self.all_path_lens['value'].values)

    def mean_shortest_path_length(self):
        return self.all_path_lens['value'].values.mean()

    def assortativity_coefficient(self):
        return nx.degree_assortativity_coefficient(self.G_undirected)

    def number_of_connected_components(self):
        return nx.number_connected_components(self.G_undirected)

    def diameter_of_largest_connected_components(self):
        return nx.diameter(
            max(nx.connected_component_subgraphs(self.G_undirected), key=len))

    def average_clustering_coefficient(self):
        return nx.average_clustering(self.G_undirected)

    def min_node_degree(self):
        return min(dict(self.G_undirected.degree()).values())

    def max_node_degree(self):
        return max(dict(self.G_undirected.degree()).values())

    def mean_node_degree(self):
        return sum([v for x, v in self.G_undirected.degree()]) / len(self.G_undirected.degree())

    def degree_distribution(self):
        return pd.DataFrame([{'node': node, 'value': deg} for node, deg in self.G_undirected.degree()])

    def page_rank_distribution(self):
        return pd.DataFrame(
            [{"node": node, "value": pg_rank} for node, pg_rank in nx.pagerank(self.G_undirected).items()])

    def community_structure(self):
        try:
            C = community.best_partition(self.G_undirected)
            # print(list(C))
            return community.modularity(C,self.G_undirected)
        except Exception as e:
            print(e)


class GithubNetworkMeasurements(NetworkMeasurements):

    def __init__(self, project_on='content', weighted=False, **kwargs):
        self.project_on = project_on
        self.weighted = weighted
        super(GithubNetworkMeasurements, self).__init__(**kwargs)

    def preprocess(self):
        pass

    def build_undirected_graph(self, df):
        dnx = nx.from_pandas_edgelist(self.main_df, source=self.node1, target=self.node2)

        dnx.add_nodes_from(self.main_df[self.node1 if self.node2 == self.project_on else self.node2].unique(),
                           bipartite=0)
        dnx.add_nodes_from(self.main_df[self.project_on].unique(), bipartite=1)

        top_nodes = {n for n, d in dnx.nodes(data=True) if d['bipartite'] == 0}
        bottom_nodes = set(dnx) - top_nodes
        # construct weighted graph, edge weights will be stored in "weight" edge attribute
        if self.weighted:
            self.G_undirected = bipartite.weighted_projected_graph(dnx, bottom_nodes)
        else:
            self.G_undirected = bipartite.projected_graph(dnx, bottom_nodes)


class TwitterNetworkMeasurements(NetworkMeasurements):

    def __init__(self, **kwargs):
        super(TwitterNetworkMeasurements, self).__init__(**kwargs)

    def preprocess(self):
        pass

    def build_undirected_graph(self, df):
        self.main_df = self.main_df.groupby([self.node1, self.node2]).size().reset_index(name='weight')
        self.G_undirected = nx.from_pandas_edgelist(self.main_df, source=self.node1, target=self.node2,
                                                    edge_attr='weight')

class RedditNetworkMeasurements(NetworkMeasurements):

    def __init__(self, **kwargs):
        super(RedditNetworkMeasurements, self).__init__(**kwargs)

    def preprocess(self):
        pass

    def build_undirected_graph(self, df):
        self.main_df = self.main_df.groupby([self.node1, self.node2]).size().reset_index(name='weight')
        self.G_undirected = nx.from_pandas_edgelist(self.main_df, source=self.node1, target=self.node2,
                                                    edge_attr='weight')
       
def run_metrics(ground_measurement, simulation_measurement):
    from prettytable import PrettyTable

    def run(conf):
        all_results= PrettyTable()
        all_results.field_names = ["Measurement","Metric","Result"]
        for measurement_name, params in tqdm.tqdm(conf.items()):
            #print("Computing Measurement_name: {}".format(measurement_name))
            try:
                ground_result = getattr(ground_measurement, params['measurement'])()

                simulation_result = getattr(simulation_measurement, params['measurement'])()

                for metric_name, metric_fx in params['metrics'].items():
                    all_results.add_row([measurement_name,metric_name,metric_fx(ground_result, simulation_result)])

            except AttributeError as e:
                print("No method: {} Error {}".format(measurement_name,e))
                all_results.add_row([measurement_name, metric_name, "Error"])
        return all_results

    return run


if __name__ == '__main__':
    import os
    import network_config as nx_c

    path = '~/infrastructure/cve_network_triplets.csv'
    # path = '~/test.csv'
    github_nx_gold = GithubNetworkMeasurements(data=os.path.expanduser(path), node1='actor', node2='repo',
                                               project_on='repo')
    github_nx_sim = GithubNetworkMeasurements(data=os.path.expanduser(path), node1='actor', node2='repo',
                                              project_on='repo')

    results = run_metrics(ground_measurement=github_nx_gold, simulation_measurement=github_nx_sim)(
        nx_c.network_measurement_params)
    print(results)
