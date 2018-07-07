import pandas as pd
from functools import partial, update_wrapper
import Metrics
import RepoCentricMeasurements
import UserCentricMeasurements
#from load_data import load_data
from Measurements import *

import pprint


def named_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    partial_func.varnames = func.__code__.co_varnames
    return partial_func


contribution_events = ["PullRequestEvent", "PushEvent", "IssuesEvent","IssueCommentEvent","PullRequestReviewCommentEvent","CommitCommentEvent","CreateEvent"]
popularity_events = ["WatchEvent", "ForkEvent"]

measurement_params = {
    ### User Centric Measurements
     "user_unique_repos": {
         'question': '17',
         "scale": "population",
         "node_type":"user",
         "measurement": "getUserUniqueRepos",
         "measurement_args":{"eventType":contribution_events},
         "metrics": {
             "js_divergence": named_partial(Metrics.js_divergence, discrete=False),
             "rmse": Metrics.rmse,
             "r2": Metrics.r2}
     },

     "user_activity_timeline": {
         "question": '19',
         "scale": "node",
         "node_type":"user",
         "measurement": "getUserActivityTimeline",
         "measurement_args":{"eventType":contribution_events},
         "metrics": {"rmse": Metrics.rmse,
                     "ks_test": Metrics.ks_test,
                     "dtw": Metrics.dtw}
    
     },

     "user_activity_distribution": {
         "question": '24a',
         "scale": "population",
         "node_type":"user",
         "measurement": "getUserActivityDistribution",
         "measurement_args":{"eventType":contribution_events + popularity_events},
         "metrics": {"rmse": Metrics.rmse,
                     "r2": Metrics.r2,
                     "js_divergence": named_partial(Metrics.js_divergence, discrete=True)}
     },
    
     "most_active_users": {
         "question": '24b',
         "scale": "population",
         "node_type":"user",
         "measurement": "getMostActiveUsers",
         "measurement_args":{"eventType":contribution_events + popularity_events},
         "metrics": {"rbo": named_partial(Metrics.rbo_score, p=0.999)}
     },

     "user_popularity": {
         "question": '25',
         "scale": "population",
         "node_type":"user",
         "measurement": "getUserPopularity",
         "measurement_args":{"eventType":popularity_events + ['CreateEvent']},
         "metrics": {"rbo": named_partial(Metrics.rbo_score, p=0.999)}
     },

     "user_gini_coef": {
         "question": '26a',
         "scale": "population",
         "node_type":"user",
         "measurement": "getGiniCoef",
         "measurement_args":{"nodeType":"user","eventType":contribution_events},
         "metrics": {"absolute difference": Metrics.absolute_difference}
     },

     "user_palma_coef": {
         "question": '26b',
         "scale": "population",
         "node_type":"user",
         "measurement": "getPalmaCoef",
         "measurement_args":{"nodeType":"user","eventType":contribution_events},
         "metrics": {"absolute difference": Metrics.absolute_difference}
     },

     "user_diffusion_delay": {
         "question": '27',
         "scale": "population",
         "node_type":"user",
         "measurement": "getUserDiffusionDelay",
         "measurement_args":{"eventType":contribution_events},
         "metrics": {"ks_test": Metrics.ks_test}
     },

     "user_trustingness":{
         "question": '29',
         "scale": "population",
         "node_type":"user",
         "measurement":"getUserPullRequestAcceptance",
         "measurement_args":{"eventType":["PullRequestEvent"]},
         "metrics":{"ks_test":Metrics.ks_test}
         }
}

repo_measurement_params = {
    ##Repo-centric measurements
     "repo_diffusion_delay": {
         "question": 1,
         "scale": "node",
         "node_type":"repo",
         "measurement": "getRepoDiffusionDelay",
         "measurement_args":{"eventType":popularity_events},
         "metrics": {"ks_test": Metrics.ks_test,
                     "js_divergence": named_partial(Metrics.js_divergence, discrete=False)},
     },
   
     "repo_growth": {
         "question": 2,
         "scale": "node",
         "node_type":"repo",
         "measurement": "getRepoGrowth",
         "measurement_args":{"eventType":contribution_events},
         "metrics": {"rmse": named_partial(Metrics.rmse, join="outer"),
                     "dtw": Metrics.dtw}
     },
   
     "repo_contributors": {
         "question": 4,
         "scale": "node",
         "node_type":"repo",
         "measurement": "getContributions",
         "measurement_args":{"eventType":contribution_events},
         "metrics": {"rmse": named_partial(Metrics.rmse, join="outer"),
                     "dtw": Metrics.dtw}
     },
   
    "repo_event_distribution_daily": {
        "question": 5,
        "scale": "node",
        "node_type":"repo",
       "measurement": "getDistributionOfEvents",
        "metrics": {"js_divergence": named_partial(Metrics.js_divergence, discrete=True)}
    },
   
    "repo_event_distribution_dayofweek": {
        "question": 5,
        "scale": "node",
        "node_type":"repo",
        "measurement": "getDistributionOfEvents",
        "measurement_args":{"weekday":True},
        "metrics": {"js_divergence": named_partial(Metrics.js_divergence, discrete=True)}
    },
   
     "repo_popularity_distribution": {
         "question": 12,
         "scale": "population",
         "node_type":"repo",
         "measurement": "getDistributionOfEventsByRepo",
         "measurement_args":{"eventType":["WatchEvent"]},
         "metrics": {"js_divergence": named_partial(Metrics.js_divergence, discrete=False),
                     "rmse": Metrics.rmse,
                     "r2": Metrics.r2}
     },
    
     "repo_popularity_topk": {
         "question": 12,
         "scale": "population",
         "node_type":"repo",
         "measurement": "getTopKRepos",
         "measurement_args":{"k":5000,"eventType":["WatchEvent"]},
         "metrics": {"rbo": named_partial(Metrics.rbo_score, p=0.999)}
     },
    
     "repo_liveliness_distribution": {
         "question": 13,
         "scale": "population",
         "node_type":"repo",
         "measurement": "getDistributionOfEventsByRepo",
         "measurement_args":{"eventType":["ForkEvent"]},
         "metrics": {"js_divergence": named_partial(Metrics.js_divergence, discrete=False),
                     "rmse": Metrics.rmse,
                     "r2": Metrics.r2}
     },
    
     "repo_liveliness_topk": {
         "question": 13,
         "scale": "population",
         "node_type":"repo",
         "measurement": "getTopKRepos",
         "measurement_args":{"k":5000,"eventType":["ForkEvent"]},
         "metrics": {"rbo": named_partial(Metrics.rbo_score, p=0.999)}
     },
    
     "repo_activity_disparity_gini_fork": {
         "question": 14,
         "scale": "population",
         "node_type":"repo",
         "measurement": "getGiniCoef",
         "measurement_args":{"eventType":["ForkEvent"]},
         "metrics": {"absolute_difference": Metrics.absolute_difference}
     },
    
     "repo_activity_disparity_palma_fork": {
         "question": 14,
         "scale": "population",
         "node_type":"repo",
         "measurement": "getPalmaCoef",
         "measurement_args":{"eventType":["ForkEvent"]},
         "metrics": {"absolute_difference": Metrics.absolute_difference}
     },
    
     "repo_activity_disparity_gini_push": {
         "question": 14,
         "scale": "population",
         "node_type":"repo",
         "filters": {"event": ["PushEvent"]},
         "measurement": "getGiniCoef",
         "measurement_args":{"eventType":["PushEvent"]},
         "metrics": {"absolute_difference": Metrics.absolute_difference}
     },
    
     "repo_activity_disparity_palma_push": {
         "question": 14,
         "scale": "population",
         "node_type":"repo",
         "measurement": "getPalmaCoef",
         "measurement_args":{"eventType":["PushEvent"]},
         "metrics": {"absolute_difference": Metrics.absolute_difference}
     },
    
     "repo_activity_disparity_gini_pullrequest": {
         "question": 14,
         "scale": "population",
         "node_type":"repo",
         "measurement": "getGiniCoef",
         "measurement_args":{"eventType":["PullRequestEvent"]},
         "metrics": {"absolute_difference": Metrics.absolute_difference}
     },
    
     "repo_activity_disparity_palma_pullrequest": {
         "question": 14,
         "scale": "population",
         "node_type":"repo",
         "measurement_args":{"eventType":["PullRequestEvent"]},
         "measurement": "getPalmaCoef",
         "metrics": {"absolute_difference": Metrics.absolute_difference}
     },
    
     "repo_activity_disparity_gini_issue": {
         "question": 14,
         "scale": "population",
         "node_type":"repo",
         "measurement": "getGiniCoef",
         "measurement_args":{"eventType":["IssuesEvent"]},
         "metrics": {"absolute_difference": Metrics.absolute_difference}
     },
    
     "repo_activity_disparity_palma_issue": {
         "question": 14,
         "scale": "population",
         "node_type":"repo",
         "measurement": "getPalmaCoef",
         "measurement_args":{"eventType":["IssuesEvent"]},
         "metrics": {"absolute_difference": Metrics.absolute_difference}
     },
    
     "repo_trustingness":{
         "question": '15',
         "scale": "population",
         "node_type":"repo",
         "filters":{"event":"PullRequestEvent"},
         "measurement":"getRepoPullRequestAcceptance",
         "metrics":{"ks_test":Metrics.ks_test}
         },

    "repo_issue_to_push":{
        "question":'31',
        "scale":"node",
        "node_type":"repo",
        "measurement_args":{"eventType":contribution_events},
        "measurement":"getIssueVsPushProbability",
        "metrics":{"rmse":Metrics.rmse}
        },

     "repo_event_counts_issue": {
        "question": '11',
        "scale": "population",
        "node_type": "repo",
        "measurement": "getDistributionOfEventsByRepo",
        "measurement_args":{"eventType":["IssuesEvent"]},
        "metrics":{"js_divergence":named_partial(Metrics.js_divergence, discrete=False),
                   "rmse":Metrics.rmse,
                   "r2": Metrics.r2}
        },
     "repo_event_counts_pull_request": {
        "question": '11',
        "scale": "population",
        "node_type": "repo",
        "measurement": "getDistributionOfEventsByRepo",
        "measurement_args":{"eventType":["PullRequestEvent"]},
        "metrics":{"js_divergence":named_partial(Metrics.js_divergence, discrete=False),
                   "rmse":Metrics.rmse,
                   "r2": Metrics.r2}
        },
     "repo_event_counts_push": {
        "question": '11',
        "scale": "population",
        "node_type": "repo",
        "measurement": "getDistributionOfEventsByRepo",
        "measurement_args":{"eventType":["PushEvent"]},
        "metrics":{"js_divergence":named_partial(Metrics.js_divergence, discrete=False),
                   "rmse":Metrics.rmse,
                   "r2": Metrics.r2}
        },
     "repo_user_continue_prop":{
        "question":"30",
        "scale":"node",
        "node_type":"repo",
        "measurement":"propUserContinue",
        "measurement_args":{"eventType":contribution_events},
        "metrics":{"rmse":Metrics.rmse}
        },

#     "repo_avg_interevent_time":{
#        "question":"11",
#        "scale":"node",
#        "node_type":"repo",
#        "measurement":"getAvgTimebtwEvents",
#        "measurement_args":{"eventType":contribution_events},
#        "metrics":{"ks":Metrics.ks_test}
#        }
}

community_measurement_params = {
    #Community-level measurements
    "community_gini":{
        "question":'6',
        "scale":"community",
        "node_type":"repo",
        "measurement":"getCommunityGini",
        "measurement_args":{"eventType":contribution_events},
        "metrics":{"absolute_difference": Metrics.absolute_difference}
        },

    "community_palma":{
        "question":'6',
        "scale":"community",
        "node_type":"repo",
        "measurement":"getCommunityPalma",
        "measurement_args":{"eventType":contribution_events},
        "metrics":{"absolute_difference": Metrics.absolute_difference}
        },

    "community_geo_locations":{
        "question":'21',
        "scale":"community",
        "node_type":"user",
        "measurement":"userGeoLocation",
        "measurement_args":{"eventType":contribution_events},
        "metrics":{"js_divergence":named_partial(Metrics.js_divergence, discrete=False)}
        },

    "community_event_proportions":{
        "question":'7',
        "scale":"community",
        "node_type":"repo",
        "measurement":"getProportion",
         "measurement_args":{"eventType":contribution_events + popularity_events},
        "metrics":{"js_divergence": named_partial(Metrics.js_divergence,discrete=True)}
        },
    "community_contributing_users":{
        "question":"20",
        "scale":"community",
        "node_type":"user",
        "measurement":"contributingUsers",
        "metrics":{"absolute_difference":Metrics.absolute_difference}
        },
    "community_num_user_actions":{
        "question":"23",
        "scale":"community",
        "node_type":"user",
        "measurement":"getNumUserActions",
        "measurement_args":{"eventType":contribution_events},
        "metrics":{"rmse": named_partial(Metrics.rmse, join="outer"),
                   "dtw": Metrics.dtw,
                   "js_divergence": named_partial(Metrics.js_divergence,discrete=False)
                   }
        },
    "community_burstiness":{
        "question":"9",
        "scale":"community",
        "node_type":"repo",
        "measurement":"burstsInCommunityEvents",
        "measurement_args":{"eventType":contribution_events + popularity_events},
        "metrics":{"absolute_difference":Metrics.absolute_difference}
        },
    "community_user_burstiness":{
        "question":"",
        "scale":"community",
        "node_type":"user",
        "measurement":"getUserBurstByCommunity",
        "metrics":{'ks_test':Metrics.ks_test}
        },
    "community_issue_types":{
        "question":"8",
        "scale":"community",
        "node_type":"repo",
        "measurement":"propIssueEvent",
        "metrics":{ "rmse": named_partial(Metrics.rmse,join='outer'),
                    "js_divergence": named_partial(Metrics.js_divergence, discrete=True)}
        },
    "community_user_account_ages":{
        "question":"10",
        "scale":"community",
        "node_type":"user",
        "measurement":"ageOfAccounts",
        "metrics":{'ks_test':Metrics.ks_test}
        }
    }

te_measurement_params = {
    #Influence measurements
    "user_interactions":{
        "question":'18a1',
        "scale":"te",
        "node_type":"user",
        "measurement":"computeTEUsers",
        "metrics": {"rbo": named_partial(Metrics.rbo_for_te, idx = 0,  wt = 0.9 ,ct = 30)}
        },
    
    "user_total":{
            "question":'18a2',
            "scale":"te",
            "node_type":"user",
            "measurement":"computeTEUsers",
            "metrics": {"rbo": named_partial(Metrics.rbo_for_te, idx = 1, wt = 0.75 , ct = 10)}
            },    

    "user_event_interactions":{
            "question":'18b',
            "scale":"te",
            "node_type":"user",
            "measurement":"computeTEUserEvents",
            "metrics": {"rbo": named_partial(Metrics.rbo_for_te, idx = 0, wt = 0.9 ,ct = 25)}
            },  
    
    "repo_interactions":{
            "question":'18c1',
            "scale":"te",
            "node_type":"repo",
            "measurement":"computeTERepos",
            "metrics": {"rbo": named_partial(Metrics.rbo_for_te, idx = 0, wt = 0.9, ct = 30)}
            }, 
    
    "repo_total":{
            "question":'18c2',
            "scale":"te",
            "node_type":"repo",
            "measurement":"computeTERepos",
            "metrics": {"rbo": named_partial(Metrics.rbo_for_te, idx = 1, wt = 0.75, ct = 10)}
            }
    }


measurement_params.update(repo_measurement_params)
measurement_params.update(community_measurement_params)
measurement_params.update(te_measurement_params)


def run_metrics(ground_truth, simulation, measurement_name,measurement_on_gt=None):


    """
    Run all of the assigned metrics for a given measurement.

    Inputs:
    ground_truth - Measurements object of ground truth data
    simulation - Measurements object of simulated data
    measurement_name - Name of measurement corresponding to keys of measurement_params

    Outputs:
    measurement_on_gt - Output of the measurement for the ground truth data
    measurement_on_sim - Output of the measurement for the simulation data
    metrics_output - Dictionary containing metric results for each metric assigned to the measurement
    """


    p = measurement_params[measurement_name]
    if "measurement_args" in p:
        measurement_args = p["measurement_args"]
    else:
        measurement_args = {}


    metrics_output = {}

    #ground_truth measurement
    if measurement_on_gt is None:
        measurement_function = getattr(ground_truth,p['measurement'])
        print("Measuring {} for ground truth data".format(measurement_function.__name__))
        measurement_on_gt = measurement_function(**measurement_args)
        if p["scale"] == "te":
            measurement_on_gt = measurement_on_gt
        print measurement_function.__name__
        print measurement_on_gt

    #simulation measurement
    measurement_function = getattr(simulation,p['measurement'])
    print("Measuring {} for simulation data".format(measurement_function.__name__))
    measurement_on_sim = measurement_function(**measurement_args)
    if p["scale"] == "te":
        measurement_on_sim = measurement_on_sim
        

    metrics = p['metrics']

    if p["scale"] in ["node","community"]:
        for node in measurement_on_gt:
            metrics_output[node] = {}


    #iterate over the metrics assigned to the measurement
    for m, metric_function in metrics.items():
        print("Calculating {} for {}".format(metric_function.__name__, measurement_function.__name__))

        if p["scale"] in ["node","community"]:

            #iterate over individual nodes and communities to calculate the metric results for each
            for node in measurement_on_gt:
                
                print(node)
        
                if node in measurement_on_gt and node in measurement_on_sim:
                    print(measurement_on_gt[node])
                    if not measurement_on_gt[node] is None and not measurement_on_sim[node] is None:
                        metric = metric_function(measurement_on_gt[node],measurement_on_sim[node])
                    else:
                        metric=None
                else:
                    metric = None

                metrics_output[node][m] = metric
        else:
            metric = metric_function(measurement_on_gt, measurement_on_sim)
            metrics_output[m] = metric

    print('measurements')
    print(measurement_on_gt)
    print(measurement_on_sim)
    print('metrics_output',metrics_output)
    return measurement_on_gt, measurement_on_sim, metrics_output



def run_all_metrics(ground_truth, simulation, scale=None, node_type = None):

    """
    Calculate metrics for multiple measurements.

    Inputs:
    ground_truth - Measurements object with ground truth data or a ground truth measurement output dictionary.  If it is a Measuremetns object the meausurements will be calculated for the groud truth.  If it is a dictionary, the pre-calculated measurements will be used.
    simulation - Simulation Meausrements object
    scale = Select measurements of a particular scale, possible values are currently "node" or "population".  If None, measurements of all scales are included.
    node_type = Select measurements of particular node-type, possible values are "repo" or "user".  If None, measurements of both node types are included.
    """

    results = {}
    #select measurements of desired scale and node type
    measurements = [m for m, m_info in measurement_params.items() if (scale is None or m_info["scale"] == scale) and (node_type is None or m_info["node_type"] == node_type)]

    for measurement_name in measurements:
        gt, sim, metric_results = run_metrics(ground_truth, simulation, measurement_name)
        results[measurement_name] = metric_results
    return results


def main():

    ###READ IN ground_truth and simulation here
    #Data should be in 4-column format: time, event, user, repo
    #ground_truth, simulation = load_data(weeks=True)

    user_ids = ['RIH-7636kqldbT3q-mKVNg','RNCPDvxzygRe8m7ENWg9Kw','ZjuuEc-QjH5b4E3FtQencw','_Qc4tzHyLBsDFu-q4HpVnw']
    repo_ids = ['sG2sD5eAH3ojlZYCsX3hJg/sG2sD5eAH3ojlZYCsX3hJg','DXUQl8d5BBrhwGo5eU5d5Q/iS-SlfdKFS3N_iSpaYLX3Q',
                'x9BrCoUrzYi11O-5Y-tFzg/2c9v3EnK2YrZcVgb0shFyQ','2-scMrZv13F95YPZmfieww/1EaArWHXzf8AhyhA34CX6w']


    #instantiate Measurement objects for both the ground truth and simulation data
    gt_measurements = Measurements(ground_truth,
                                   interested_users=user_ids,
                                   interested_repos=repo_ids)
    sim_measurements = Measurements(simulation,
                                    interested_users=user_ids,
                                    interested_repos=repo_ids)


    #run individual metric
    gt_measurement, sim_measurement, metric = run_metrics(gt_measurements, sim_measurements, "user_unique_repos")
    pprint.pprint(metric)


    #run all assigned metrics
    metrics = run_all_metrics(gt_measurements,sim_measurements)
    pprint.pprint(metrics)


    #run all assigned population-level metrics
    metrics = run_all_metrics(gt_measurements,sim_measurements,scale="population")
    pprint.pprint(metrics)


    #run all assigned repo-centric metrics for the node-level measurements
    metrics = run_all_metrics(gt_measurements,sim_measurements,node_type="user",scale="node")
    pprint.pprint(metrics)


if __name__ == "__main__":
    main()
