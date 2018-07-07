import pandas as pd
from functools import partial, update_wrapper
import Metrics
import RepoCentricMeasurements
import UserCentricMeasurements
#from load_data import load_data

import pprint

def named_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


contribution_events = ["PullRequestEvent", "PushEvent", "IssuesEvent","IssueCommentEvent","PullRequestReviewComment","CommitCommentEvent","CreateEvent"]
popularity_events = ["WatchEvent", "ForkEvent"]

measurement_params = {
    ### User Centric Measurements

    "user_unique_repos": {
        'question': '17',
        "scale": "population",
        "node_type":"user",
        "filters": {"event": contribution_events},
        "measurement": UserCentricMeasurements.getUserUniqueRepos,
        "metrics": { 
            "js_divergence": named_partial(Metrics.js_divergence, discrete=False),
            "rmse": Metrics.rmse,
            "r2": Metrics.r2}
    },

    "user_activity_timeline": {
        "question": '19',
        "scale": "node",
        "node_type":"user",
        "filters": {"event": contribution_events},
        "measurement": UserCentricMeasurements.getUserActivityTimeline,
        "metrics": {"rmse": Metrics.rmse,
                    "ks_test": Metrics.ks_test,
                    "dtw": Metrics.dtw}

    },

    "user_activity_distribution": {
        "question": '24a',
        "scale": "population",
        "node_type":"user",
        "measurement": UserCentricMeasurements.getUserActivityDistribution,
        "metrics": {"rmse": Metrics.rmse,
                    "r2": Metrics.r2,
                    "js_divergence": named_partial(Metrics.js_divergence, discrete=True)}
    },

    "most_active_users": {
        "question": '24b',
        "scale": "population",
        "node_type":"user",
        "measurement": named_partial(UserCentricMeasurements.getMostActiveUsers, k=5000),
        "metrics": {"rbo": named_partial(Metrics.rbo_score, p=0.95)}
    },

    "user_popularity": {
        "question": '25',
        "scale": "population",
        "node_type":"user",
        "filters": {"event": popularity_events + ['CreateEvent']},
        "measurement": named_partial(UserCentricMeasurements.getUserPopularity, k=5000),
        "metrics": {"rbo": named_partial(Metrics.rbo_score, p=0.95)}
    },

    "user_gini_coef": {
        "question": '26a',
        "scale": "population",
        "node_type":"user",
        "filters": {"event": contribution_events},
        "measurement": UserCentricMeasurements.getGiniCoef,
        "metrics": {"absolute difference": Metrics.absolute_difference}
    },

    "user_palma_coef": {
        "question": '26b',
        "scale": "population",
        "node_type":"user",
        "filters": {"event":contribution_events},
        "measurement": UserCentricMeasurements.getPalmaCoef,
        "metrics": {"absolute difference": Metrics.absolute_difference}
    },

    "user_diffusion_delay": {
        "question": '27',
        "scale": "population",
        "node_type":"user",
        "filters": {"event": contribution_events},
        "measurement": UserCentricMeasurements.getUserDiffusionDelay,
        "metrics": {"ks_test": Metrics.ks_test}
    },

}

repo_measurement_params = {
    "repo_diffusion_delay": {
        "question": 1,
        "scale": "node",
        "node_type":"repo",
        "filters":{"event": popularity_events},
        "measurement": named_partial(RepoCentricMeasurements.getRepoDiffusionDelay,
                                     eventType=popularity_events,
                                     ),
        "metrics": {"ks_test": Metrics.ks_test,
                    "js_divergence": named_partial(Metrics.js_divergence, discrete=False)},
    },
    "repo_growth": {
        "question": 2,
        "scale": "node",
        "node_type":"repo",
        "filters": {"event": contribution_events},
        "measurement": RepoCentricMeasurements.getRepoGrowth,
        "metrics": {"rmse": named_partial(Metrics.rmse, join="outer"),
                    "dtw": Metrics.dtw}
    },
    "repo_contributors": {
        "question": 4,
        "scale": "node",
        "node_type":"repo",
        "filters": {"event": contribution_events},
        "measurement": RepoCentricMeasurements.getContributions,
        "metrics": {"rmse": named_partial(Metrics.rmse, join="outer"),
                    "dtw": Metrics.dtw}
    },
    "repo_event_distribution_daily": {
        "question": 5,
        "scale": "node",
        "node_type":"repo",
        "measurement": RepoCentricMeasurements.getDistributionOfEvents,
        "metrics": {"js_divergence": named_partial(Metrics.js_divergence, discrete=True)}
    },
    "repo_event_distribution_dayofweek": {
        "question": 5,
        "scale": "node",
        "node_type":"repo",
        "measurement": named_partial(RepoCentricMeasurements.getDistributionOfEvents, weekday=True),
        "metrics": {"js_divergence": named_partial(Metrics.js_divergence, discrete=True)}
    },
    "repo_popularity_distribution": {
        "question": 12,
        "scale": "population",
        "node_type":"repo",
        "filters": {"event": ["WatchEvent"]}, 
        "measurement": named_partial(RepoCentricMeasurements.getDistributionOfEventsByRepo, eventType='WatchEvent'),
        "metrics": {"js_divergence": named_partial(Metrics.js_divergence, discrete=False),
                    "rmse": Metrics.rmse,
                    "r2": Metrics.r2}
    },
    "repo_popularity_topk": {
        "question": 12,
        "scale": "population",
        "node_type":"repo",
        "filters": {"event": ["WatchEvent"]}, 
        "measurement": named_partial(RepoCentricMeasurements.getTopKRepos, k=5000, eventType='WatchEvent'),
        "metrics": {"rbo": named_partial(Metrics.rbo_score, p=0.95)}
    },
    "repo_liveliness_distribution": {
        "question": 13,
        "scale": "population",
        "node_type":"repo",
        "filters": {"event": ["ForkEvent"]}, 
        "measurement": named_partial(RepoCentricMeasurements.getDistributionOfEventsByRepo, eventType='ForkEvent'),
        "metrics": {"js_divergence": named_partial(Metrics.js_divergence, discrete=False),
                    "rmse": Metrics.rmse,
                    "r2": Metrics.r2}
    },
    "repo_liveliness_topk": {
        "question": 13,
        "scale": "population",
        "node_type":"repo",
        "filters": {"event": ["ForkEvent"]}, 
        "measurement": named_partial(RepoCentricMeasurements.getTopKRepos, k=5000, eventType='ForkEvent'),
        "metrics": {"rbo": named_partial(Metrics.rbo_score, p=0.95)}
    },
    "repo_activity_disparity_gini_fork": {
        "question": 14,
        "scale": "population",
        "node_type":"repo",
        "filters": {"event": ["ForkEvent"]},
        "measurement": RepoCentricMeasurements.getGiniCoef,
        "metrics": {"absolute_difference": Metrics.absolute_difference}
    },
    "repo_activity_disparity_palma_fork": {
        "question": 14,
        "scale": "population",
        "node_type":"repo",
        "filters": {"event": ["ForkEvent"]},
        "measurement": RepoCentricMeasurements.getPalmaCoef,
        "metrics": {"absolute_difference": Metrics.absolute_difference}
    },
    "repo_activity_disparity_gini_push": {
        "question": 14,
        "scale": "population",
        "node_type":"repo",
        "filters": {"event": ["PushEvent"]},
        "measurement": RepoCentricMeasurements.getGiniCoef,
        "metrics": {"absolute_difference": Metrics.absolute_difference}
    },
    "repo_activity_disparity_palma_push": {
        "question": 14,
        "scale": "population",
        "node_type":"repo",
        "filters": {"event": ["PushEvent"]},
        "measurement": RepoCentricMeasurements.getPalmaCoef,
        "metrics": {"absolute_difference": Metrics.absolute_difference}
    },
    "repo_activity_disparity_gini_pullrequest": {
        "question": 14,
        "scale": "population",
        "node_type":"repo",
        "filters": {"event": ["PullRequestEvent"]},
        "measurement": RepoCentricMeasurements.getGiniCoef,
        "metrics": {"absolute_difference": Metrics.absolute_difference}
    },
    "repo_activity_disparity_palma_pullrequest": {
        "question": 14,
        "scale": "population",
        "node_type":"repo",
        "filters": {"event": ["PullRequestEvent"]},
        "measurement": RepoCentricMeasurements.getPalmaCoef,
        "metrics": {"absolute_difference": Metrics.absolute_difference}
    },
    "repo_activity_disparity_gini_issue": {
        "question": 14,
        "scale": "population",
        "node_type":"repo",
        "filters": {"event": ["IssuesEvent"]},
        "measurement": RepoCentricMeasurements.getGiniCoef,
        "metrics": {"absolute_difference": Metrics.absolute_difference}
    },
    "repo_activity_disparity_palma_issue": {
        "question": 14,
        "scale": "population",
        "node_type":"repo",
        "filters": {"event": ["IssuesEvent"]},
        "measurement": RepoCentricMeasurements.getPalmaCoef,
        "metrics": {"absolute_difference": Metrics.absolute_difference}
    }
}

measurement_params.update(repo_measurement_params)


def prefilter(data, filters):

    """
    Filter the data frame based on a set of fields and values.  
    Used to subset on specific event types and on specific users and repos for node-level measurements

    Inputs:
    data - DataFrame in 4-column format
    filters - A dictionary with keys indicating the field to filter, and values indicating the values of the field to keep

    Outputs:
    data - The filtered data frame

    """

    data.columns = ['time', 'event', 'user', 'repo']
    for field, values in filters.items():
        data = data[data[field].isin(values)]
    return data


def run_metrics(ground_truth, simulation, measurement_name,users=None,repos=None):

    """
    Run all of the assigned metrics for a given measurement.

    Inputs:
    ground_truth - DataFrame of ground truth data
    simulation - DataFrame of simulated data
    measurement_name - Name of measurement corresponding to keys of measurement_params
    users - list of user IDs for user-centric, node-level measurements
    repos - list of repo IDs for repo-centric, node-level measurements

    Outputs:
    measurement_on_gt - Output of the measurement for the ground truth data
    measurement_on_sim - Output of the measurement for the simulation data
    metrics_output - Dictionary containing metric results for each metric assigned to the measurement   
    """

    p = measurement_params[measurement_name]

    if p["node_type"] == "user":
        nodes = users
    else:
        nodes = repos


    if "filters" in p:
        ground_truth = prefilter(ground_truth, p['filters'])
        simulation = prefilter(simulation, p['filters'])


    #for node-level measurements default to the most active node if a 
    #list of nodes is not provided
    if p["scale"] == "node" and nodes is None:
        nodes = ground_truth.groupby([p["node_type"],'event'])["time"].count().reset_index()
        nodes = nodes.groupby(p["node_type"])["time"].median().sort_values(ascending=False).reset_index()
        nodes = nodes.head(1)[p["node_type"]]
    elif p["scale"] != "node":
        nodes = ['']


    metrics_output = {}
 
    #for node level measurements iterate over nodes
    for node in nodes:
        
        if p["scale"] == "node":

            metrics_output[node] = {}

            #select data for individual node
            filter = {p["node_type"]:[node]}
            gt = prefilter(ground_truth, filter)
            sim = prefilter(simulation, filter)
        else:
            gt = ground_truth.copy()
            sim = simulation.copy()
            

        measurement_function = p['measurement']

        empty_df = False
        if len(gt.index) > 0:
            print("Measuring {} for ground truth data".format(measurement_function.__name__))
            measurement_on_gt = measurement_function(gt)
        else:
            print("Ground truth data frame is empty for {} measurement".format(measurement_function.__name__))
            empty_df = True
            measurement_on_gt = []

        if len(sim.index) > 0:
            print("Measuring {} for simulation data".format(measurement_function.__name__))
            measurement_on_sim = measurement_function(sim)
        else:
            print("Simulation data frame is empty for {} measurement".format(measurement_function.__name__))
            empty_df = True
            measurement_on_sim = []


        metrics = p['metrics']

        #iterate over the metrics assigned to the measurement
        for m, metric_function in metrics.items():
            print("Calculating {} for {}".format(metric_function.__name__, measurement_function.__name__))
            if not empty_df:
                metric = metric_function(measurement_on_gt, measurement_on_sim)
            else:
                metric = None

            if p["scale"] == "node":
                metrics_output[node][m] = metric
            else:
                metrics_output[m] = metric
                

    return measurement_on_gt, measurement_on_sim, metrics_output


def run_all_metrics(ground_truth, simulation, scale=None, node_type = None, users = None, repos = None):

    """
    Calculate metrics for multiple measurements.

    Inputs:
    ground_truth - Ground truth data frame
    simulation - Simulation data frame
    scale = Select measurements of a particular scale, possible values are currently "node" or "population".  If None, measurements of all scales are included.
    node_type = Select measurements of particular node-type, possible values are "repo" or "user".  If None, measurements of both node types are included.
    users = List of users to use for node-level user measurements.  If None, the most active user from the ground truth data is selected for all user node-level measurements
    repos = List of repos to user for node-level repo measurements. If None, the most active repo from the ground truth data is selected for all repo node-level measurements
    """

    results = {}
    #select measurements of desired scale and node type
    measurements = [m for m, m_info in measurement_params.items() if (scale is None or m_info["scale"] == scale) and (node_type is None or m_info["node_type"] == node_type)] 

    for measurement_name in measurements:
        gt, sim, metric_results = run_metrics(ground_truth.copy(), simulation.copy(), measurement_name, users=users, repos=repos)
        results[measurement_name] = metric_results
    return results


def main():

    ###READ IN ground_truth and simulation here
    #Data should be in 4-column format: time, event, user, repo
    #ground_truth, simulation = load_data()

    ground_truth.columns = ['time','event','user','repo']
    simulation.columns = ['time','event','user','repo']


    #run individual metric
    gt_measurement, sim_measurement, metric = run_metrics(ground_truth, simulation, "repo_contributors")
    pprint.pprint(metric)


    #run individual metric
    gt_measurement, sim_measurement, metric = run_metrics(ground_truth, simulation, "user_activity_timeline")
    pprint.pprint(metric)


    #run individual metric for specific users for the node-level measurement
    gt_measurement, sim_measurement, metric = run_metrics(ground_truth, simulation, "user_activity_timeline",users=['PeZv4Yha0B_17dV8SAioFA'])
    pprint.pprint(metric)


    #run all assigned metrics
    metrics = run_all_metrics(ground_truth,simulation)
    pprint.pprint(metrics)


    #run all assigned population-level metrics 
    metrics = run_all_metrics(ground_truth,simulation,scale="population")
    pprint.pprint(metrics)


    #run all assigned repo-centric metrics with specific nodes for the node-level measurements
    metrics = run_all_metrics(ground_truth,simulation,node_type="repo",repos=['3mGSybhub0IE-iZ0nOcOmg/fxFnpSLfvseMwBr1Z3NPkw','B8ZJ9zQBfx4zJyuG6QCWcQ/73uOGPnes5YM9RW6Bst3GQ'])
    pprint.pprint(metrics)


if __name__ == "__main__":
    main()
