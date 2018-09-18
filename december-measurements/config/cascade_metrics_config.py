import Metrics


from functools import partial, update_wrapper
def named_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    partial_func.varnames = func.__code__.co_varnames
    return partial_func


def get_node_level_measurements_params(time_granularity):
    """
    Return params dictionary for node level (i.e. cascade) measurements
    for the given cascade passed by root_id
    :param time_granularity: "Y", "M", "D", "H" [years/months/days/hours] for timeseries measurements
    :return: node_level_measurement_params
    """
    # ["depth", "breadth", "size", "structural_virality", "unique_nodes", "new_node_ratio"])
    node_level_measurement_params = {
        "cascade_max_depth_over_time": {
            "scale":"node",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":False,
            "measurement":"cascade_timeseries_of",
            "measurement_args":{"attribute": "depth", "time_granularity": time_granularity},
            "metrics":{"rmse": Metrics.rmse,
                       "r2": Metrics.r2},
            "temporal_vs_batch":"Temporal"
        },
        "cascade_breadth_by_time": {
            "scale":"node",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":False,
            "measurement":"cascade_timeseries_of",
            "measurement_args":{"attribute": "breadth", "time_granularity": time_granularity},
            "metrics":{"rmse": Metrics.rmse,
                       "r2": Metrics.r2},
            "temporal_vs_batch":"Temporal"
        },
        "cascade_structural_virality_over_time": {
            "scale":"node",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":False,
            "measurement":"cascade_timeseries_of",
            "measurement_args":{"attribute": "structural_virality", "time_granularity": time_granularity},
            "metrics":{"rmse": Metrics.rmse,
                       "r2": Metrics.r2},
            "temporal_vs_batch":"Temporal"
        },
        "cascade_size_over_time": {
            "scale":"node",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":False,
            "measurement":"cascade_timeseries_of",
            "measurement_args":{"attribute": "size", "time_granularity": time_granularity},
            "metrics":{"rmse": Metrics.rmse,
                       "r2": Metrics.r2},
            "temporal_vs_batch":"Temporal"
        },
        "cascade_uniq_users_by_time": {
            "scale": "node",
            "node_type": "Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":False,
            "measurement": "cascade_timeseries_of",
            "measurement_args": {"attribute": "unique_nodes", "time_granularity": time_granularity},
            "metrics": {"rmse": Metrics.rmse,
                        "r2": Metrics.r2},
            "temporal_vs_batch": "Temporal"
        },
        "cascade_new_user_ratio_by_time": {
            "scale": "node",
            "node_type": "Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":False,
            "measurement": "cascade_timeseries_of",
            "measurement_args": {"attribute": "new_node_ratio", "time_granularity": time_granularity},
            "metrics": {"rmse": Metrics.rmse,
                        "r2": Metrics.r2},
            "temporal_vs_batch": "Temporal"
        },
        "cascade_breadth_by_depth": {
            "scale":"node",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":False,
            "measurement":"cascade_depth_by",
            "measurement_args":{"attribute": "breadth"},
            "metrics":{"rmse": Metrics.rmse,
                       "r2": Metrics.r2},
            "temporal_vs_batch":"Batch"
        },
        "cascade_new_user_ratio_by_depth": {
            "scale":"node",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":False,
            "measurement":"cascade_depth_by",
            "measurement_args":{"attribute": "new_node_ratio"},
            "metrics":{"rmse": Metrics.rmse,
                       "r2": Metrics.r2},
            "temporal_vs_batch":"Batch"
        },
        "cascade_uniq_users_by_depth": {
            "scale":"node",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":False,
            "measurement":"cascade_depth_by",
            "measurement_args":{"attribute": "unique_nodes"},
            "metrics":{"rmse": Metrics.rmse,
                       "r2": Metrics.r2},
            "temporal_vs_batch":"Batch"
        },
        "cascade_participation_gini": {
            "scale":"node",
            "node_type":"Cascade",
            "measurement":"cascade_participation_gini",
            "measurement_args":{},
            "metrics":{"percentage difference": Metrics.absolute_percentage_error},
            "temporal_vs_batch":"Batch"
        },
        "cascade_participation_palma": {
            "scale":"node",
            "node_type":"Cascade",
            "measurement":"cascade_participation_palma",
            "measurement_args":{},
            "metrics":{"percentage difference": Metrics.absolute_percentage_error},
            "temporal_vs_batch":"Batch"
        }
    }
    return node_level_measurement_params


def get_community_level_measurements_params(time_granularity="M"):
    """
    Return params dictionary for community (i.e. subreddit) measurements
    for the given community passed by community_id
    :param time_granularity: "Y", "M", "D", "H" [years/months/days/hours] for timeseries measurements
    :return: community_measurement_params
    """
    community_measurement_params = {
        "community_max_depth_distribution": {
            "scale":"community",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement":"cascade_collection_distribution_of",
            "measurement_args":{"attribute": "depth"},
            "metrics":{"js_divergence": named_partial(Metrics.js_divergence, discrete=False)},
            "temporal_vs_batch":"Batch"
        },
        "community_max_breadth_distribution": {
            "scale":"community",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement":"cascade_collection_distribution_of",
            "measurement_args":{"attribute": "breadth"},
            "metrics":{"js_divergence": named_partial(Metrics.js_divergence, discrete=False)},
            "temporal_vs_batch":"Batch"
        },
        "community_structural_virality_distribution": {
            "scale":"community",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement":"cascade_collection_distribution_of",
            "measurement_args":{"attribute": "structural_virality"},
            "metrics":{"js_divergence": named_partial(Metrics.js_divergence, discrete=False)},
            "temporal_vs_batch":"Batch"
        },
        "community_cascade_size_distribution": {
            "scale":"community",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement":"cascade_collection_distribution_of",
            "measurement_args":{"attribute": "size"},
            "metrics":{"js_divergence": named_partial(Metrics.js_divergence, discrete=False)},
            "temporal_vs_batch":"Batch"
        },
        "community_cascade_lifetime_distribution": {
            "scale":"community",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement":"cascade_collection_distribution_of",
            "measurement_args":{"attribute": "lifetime"},
            "metrics":{"rmse": Metrics.rmse,
                       "r2": Metrics.r2},
            "temporal_vs_batch":"Temporal"
        },
        "community_cascade_size_timeseries": {
            "scale":"community",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement":"get_cascade_collection_size_timeseries",
            "measurement_args":{"community_grouper": "communityID", "time_granularity": time_granularity},
            "metrics":{"rmse": Metrics.rmse,
                       "r2": Metrics.r2},
            "temporal_vs_batch":"Temporal"
        },
        "community_cascade_lifetime_timeseries": {
            "scale":"community",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement":"get_cascade_collection_timeline_timeseries",
            "measurement_args":{"community_grouper": "communityID", "time_granularity": time_granularity},
            "metrics":{"js_divergence": named_partial(Metrics.js_divergence, discrete=False)},
            "temporal_vs_batch":"Batch"
        },
        "community_unique_users_by_time": {
            "scale":"community",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement":"community_users_count",
            "measurement_args":{"attribute": "unique_users", "community_grouper": "communityID", "time_granularity": time_granularity},
            "metrics":{"rmse": Metrics.rmse,
                       "r2": Metrics.r2},
            "temporal_vs_batch":"Temporal"
        },
        "community_new_user_ratio_by_time": {
            "scale":"community",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement":"get_community_users_count",
            "measurement_args":{"attribute": "new_user_ratio", "community_grouper": "communityID", "time_granularity": time_granularity},
            "metrics":{"rmse": Metrics.rmse,
                       "r2": Metrics.r2},
            "temporal_vs_batch":"Temporal"
        },
        "community_cascade_initialization_gini": {
            "scale":"community",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement":"cascade_collection_initialization_gini",
            "measurement_args":{},
            "metrics":{"percentage difference": Metrics.absolute_percentage_error},
            "temporal_vs_batch":"Batch"
        },
        "community_cascade_initialization_palma": {
            "scale":"community",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement":"cascade_collection_initialization_palma",
            "measurement_args":{},
            "metrics":{"percentage difference": Metrics.absolute_percentage_error},
            "temporal_vs_batch":"Batch"
        },
        "community_cascade_participation_gini": {
            "scale":"community",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement":"cascade_collection_participation_gini",
            "measurement_args":{},
            "metrics":{"percentage difference": Metrics.absolute_percentage_error},
            "temporal_vs_batch":"Batch"
        },
        "community_cascade_participation_palma": {
            "scale":"community",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement":"cascade_collection_participation_palma",
            "measurement_args":{},
            "metrics":{"percentage difference": Metrics.absolute_percentage_error},
            "temporal_vs_batch":"Batch"
        },
        "community_fraction_nodes_lcc": {
            "scale":"community",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement":"fraction_of_nodes_in_lcc",
            "measurement_args":{},
            "metrics":{"percentage difference": Metrics.absolute_percentage_error},
            "temporal_vs_batch":"Batch"
        }
    }
    return community_measurement_params


def get_population_level_measurements_params(time_granularity="M"):
    """
    Return params dictionary for population (i.e. all subreddits) measurements
    :param time_granularity: "Y", "M", "D", "H" [years/months/days/hours] for timeseries measurements
    :return: population_measurement_params
    """
    population_measurement_params = {
        "population_max_depth_distribution": {
            "scale":"population",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement": "cascade_collection_distribution_of",
            "measurement_args": {"attribute": "depth"},
            "metrics":{"js_divergence": named_partial(Metrics.js_divergence, discrete=False)},
            "temporal_vs_batch":"Batch"
        },
        "population_max_breadth_distribution": {
            "scale":"population",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement": "cascade_collection_distribution_of",
            "measurement_args": {"attribute": "breadth"},
            "metrics":{"js_divergence": named_partial(Metrics.js_divergence, discrete=False)},
            "temporal_vs_batch":"Batch"
        },
        "population_structural_virality_distribution": {
            "scale":"population",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement": "cascade_collection_distribution_of",
            "measurement_args": {"attribute": "structural_virality"},
            "metrics":{"js_divergence": named_partial(Metrics.js_divergence, discrete=False)},
            "temporal_vs_batch":"Batch"
        },
        "population_cascade_size_distribution": {
            "scale":"population",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement": "cascade_collection_distribution_of",
            "measurement_args": {"attribute": "size"},
            "metrics":{"js_divergence": named_partial(Metrics.js_divergence, discrete=False)},
            "temporal_vs_batch":"Batch"
        },
        "population_cascade_lifetime_distribution": {
            "scale":"population",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement": "cascade_collection_distribution_of",
            "measurement_args": {"attribute": "lifetime"},
            "metrics":{"js_divergence": named_partial(Metrics.js_divergence, discrete=False)},
            "temporal_vs_batch":"Batch"
        },
        "population_cascade_size_timeseries": {
            "scale":"population",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement":"get_cascade_collection_size_timeseries",
            "measurement_args":{"time_granularity": time_granularity},
            "metrics":{"rmse": Metrics.rmse,
                       "r2": Metrics.r2},
            "temporal_vs_batch":"Temporal"
        },
        "population_cascade_lifetime_timeseries": {
            "scale":"population",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement":"get_cascade_collection_lifetime_timeseries",
            "measurement_args":{"time_granularity": time_granularity},
            "metrics":{"rmse": Metrics.rmse,
                       "r2": Metrics.r2},
            "temporal_vs_batch":"Temporal"
        },
        "population_cascade_initialization_gini": {
            "scale":"population",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement":"cascade_collection_initialization_gini",
            "measurement_args":{ },
            "metrics":{"percentage difference": Metrics.absolute_percentage_error},
            "temporal_vs_batch":"Batch"
        },
        "population_cascade_initialization_palma": {
            "scale":"population",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement":"cascade_collection_initialization_palma",
            "measurement_args":{ },
            "metrics":{"percentage difference": Metrics.absolute_percentage_error},
            "temporal_vs_batch":"Batch"
        },
        "population_cascade_participation_gini": {
            "scale":"population",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement":"cascade_collection_participation_gini",
            "measurement_args":{ },
            "metrics":{"percentage difference": Metrics.absolute_percentage_error},
            "temporal_vs_batch":"Batch"
        },
        "population_cascade_participation_palma": {
            "scale":"population",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement":"cascade_collection_participation_palma",
            "measurement_args":{ },
            "metrics":{"percentage difference": Metrics.absolute_percentage_error},
            "temporal_vs_batch":"Batch"
        },
        "population_fraction_nodes_lcc": {
            "scale":"population",
            "node_type":"Cascade",
            "scenario1":False,
            "scenario2":True,
            "scenario3":True,
            "measurement":"fraction_of_nodes_in_lcc",
            "measurement_args":{},
            "metrics":{"percentage difference": Metrics.absolute_percentage_error},
            "temporal_vs_batch":"Batch"
        }
    }
    return population_measurement_params




cascade_node_params = get_node_level_measurements_params('D')
cascade_community_params = get_community_level_measurements_params('D')
cascade_population_params = get_population_level_measurements_params('D')

cascade_measurement_params = {}
cascade_measurement_params.update(cascade_node_params)
cascade_measurement_params.update(cascade_community_params)
cascade_measurement_params.update(cascade_population_params) 
