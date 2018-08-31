from functools import partial, update_wrapper
import Metrics
import ContentCentricMeasurements
import UserCentricMeasurements
#from load_data import load_data
from BaselineMeasurements import *

import pprint


def named_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    partial_func.varnames = func.__code__.co_varnames
    return partial_func


contribution_events = ["PullRequestEvent", "PushEvent", "IssuesEvent","IssueCommentEvent",
                       "PullRequestReviewCommentEvent","CommitCommentEvent","CreateEvent"]
popularity_events = ["WatchEvent", "ForkEvent"]

user_measurement_params = {
    ### User Centric Measurements
     "user_unique_content": {
         'question': '17',
         "scale": "population",
         "node_type":"user",
         'scenario1':True,
         'scenario2':False,
         'scenario3':True,
         "measurement": "getUserUniqueContent",
         "measurement_args":{"eventTypes":contribution_events,"content_field":"content"},
         "metrics": {
             "js_divergence": named_partial(Metrics.js_divergence, discrete=False),
             "rmse": Metrics.rmse,
             "r2": Metrics.r2}
     },

     "user_activity_timeline": {
         "question": '19',
         "scale": "node",
         "node_type":"user",
         'scenario1':False,
         'scenario2':False,
         'scenario3':False,
         "measurement": "getUserActivityTimeline",
         "measurement_args":{"eventTypes":contribution_events},
         "metrics": {"rmse": Metrics.rmse,
                     "ks_test": Metrics.ks_test,
                     "dtw": Metrics.dtw}
    
     },

     "user_activity_distribution": {
         "question": '24a',
         "scale": "population",
         "node_type":"user",
         'scenario1':True,
         'scenario2':False,
         'scenario3':True,
         "measurement": "getUserActivityDistribution",
         "measurement_args":{"eventTypes":contribution_events + popularity_events},
         "metrics": {"rmse": Metrics.rmse,
                     "r2": Metrics.r2,
                     "js_divergence": named_partial(Metrics.js_divergence, discrete=True)}
     },
    
     "most_active_users": {
         "question": '24b',
         "scale": "population",
         "node_type":"user",
         'scenario1':True,
         'scenario2':False,
         'scenario3':True,
         "measurement": "getMostActiveUsers",
         "measurement_args":{"eventTypes":contribution_events + popularity_events},
         "metrics": {"rbo": named_partial(Metrics.rbo_score, p=0.999)}
     },

     "user_popularity": {
         "question": '25',
         "scale": "population",
         "node_type":"user",
         'scenario1':True,
         'scenario2':False,
         'scenario3':True,
         "measurement": "getUserPopularity",
         "measurement_args":{"eventTypes":popularity_events + ['CreateEvent'],"content_field":"content"},
         "metrics": {"rbo": named_partial(Metrics.rbo_score, p=0.999)}
     },

     "user_gini_coef": {
         "question": '26a',
         "scale": "population",
         "node_type":"user",
         'scenario1':True,
         'scenario2':False,
         'scenario3':True,
         "measurement": "getGiniCoef",
         "measurement_args":{"nodeType":"user","eventTypes":contribution_events,"content_field":"content"},
         "metrics": {"absolute difference": Metrics.absolute_difference}
     },

     "user_palma_coef": {
         "question": '26b',
         "scale": "population",
         "node_type":"user",
         'scenario1':True,
         'scenario2':False,
         'scenario3':True,
         "measurement": "getPalmaCoef",
         "measurement_args":{"nodeType":"user","eventTypes":contribution_events,"content_field":"content"},
         "metrics": {"absolute difference": Metrics.absolute_difference}
     },

     "user_diffusion_delay": {
         "question": '27',
         "scale": "population",
         "node_type":"user",
         'scenario1':True,
         'scenario2':False,
         'scenario3':True,
         "measurement": "getUserDiffusionDelay",
         "measurement_args":{"eventTypes":contribution_events},
         "metrics": {"ks_test": Metrics.ks_test}
     },

     "user_trustingness":{
         "question": '29',
         "scale": "population",
         "node_type":"user",
         'scenario1':True,
         'scenario2':False,
         'scenario3':True,
         "measurement":"getUserPullRequestAcceptance",
         "measurement_args":{"eventTypes":["PullRequestEvent"]},
         "metrics":{"ks_test":Metrics.ks_test}
         }
}

content_measurement_params = {
    ##Content-centric measurements
     "content_diffusion_delay": {
         "question": 1,
         "scale": "node",
         "node_type":"content",
         'scenario1':False,
         'scenario2':False,
         'scenario3':False,
         "measurement": "getContentDiffusionDelay",
         "measurement_args":{"eventTypes":popularity_events,"content_field":"content","time_bin":"h"},
         "metrics": {"ks_test": Metrics.ks_test,
                     "js_divergence": named_partial(Metrics.js_divergence, discrete=False)},
     },
   
     "content_growth": {
         "question": 2,
         "scale": "node",
         "node_type":"content",
         'scenario1':False,
         'scenario2':False,
         'scenario3':False,
         "measurement": "getContentGrowth",
         "measurement_args":{"eventTypes":contribution_events,"content_field":"content"},
         "metrics": {"rmse": named_partial(Metrics.rmse, join="outer"),
                     "dtw": Metrics.dtw}
     },
   
     "content_contributors": {
         "question": 4,
         "scale": "node",
         "node_type":"content",
         'scenario1':False,
         'scenario2':False,
         'scenario3':False,
         "measurement": "getContributions",
         "measurement_args":{"eventTypes":contribution_events,"content_field":"content"},
         "metrics": {"rmse": named_partial(Metrics.rmse, join="outer"),
                     "dtw": Metrics.dtw}
     },
   
    "content_event_distribution_daily": {
        "question": 5,
        "scale": "node",
        'scenario1':False,
        'scenario2':False,
        'scenario3':False,
        "node_type":"content",
       "measurement": "getDistributionOfEvents",
         "measurement_args":{"content_field":"content"},
        "metrics": {"js_divergence": named_partial(Metrics.js_divergence, discrete=True)}
    },
   
    "content_event_distribution_dayofweek": {
        "question": 5,
        "scale": "node",
        'scenario1':False,
        'scenario2':False,
        'scenario3':False,
        "node_type":"content",
        "measurement": "getDistributionOfEvents",
        "measurement_args":{"weekday":True,"content_field":"content"},
        "metrics": {"js_divergence": named_partial(Metrics.js_divergence, discrete=True)}
    },
   
     "content_popularity_distribution": {
         "question": 12,
         "scale": "population",
         "node_type":"content",
         'scenario1':True,
         'scenario2':False,
         'scenario3':True,
         "measurement": "getDistributionOfEventsByContent",
         "measurement_args":{"eventTypes":["WatchEvent"],"content_field":"content"},
         "metrics": {"js_divergence": named_partial(Metrics.js_divergence, discrete=False),
                     "rmse": Metrics.rmse,
                     "r2": Metrics.r2}
     },
    
     "content_popularity_topk": {
         "question": 12,
         "scale": "population",
         "node_type":"content",
         'scenario1':True,
         'scenario2':False,
         'scenario3':True,
         "measurement": "getTopKContent",
         "measurement_args":{"k":5000,"eventTypes":["WatchEvent"],"content_field":"content"},
         "metrics": {"rbo": named_partial(Metrics.rbo_score, p=0.999)}
     },
    
     "content_liveliness_distribution": {
         "question": 13,
         "scale": "population",
         "node_type":"content",
         'scenario1':True,
         'scenario2':False,
         'scenario3':True,
         "measurement": "getDistributionOfEventsByContent",
         "measurement_args":{"eventTypes":["ForkEvent"],"content_field":"content"},
         "metrics": {"js_divergence": named_partial(Metrics.js_divergence, discrete=False),
                     "rmse": Metrics.rmse,
                     "r2": Metrics.r2}
     },
    
     "content_liveliness_topk": {
         "question": 13,
         "scale": "population",
         "node_type":"content",
         'scenario1':True,
         'scenario2':False,
         'scenario3':True,
         "measurement": "getTopKContent",
         "measurement_args":{"k":5000,"eventTypes":["ForkEvent"],"content_field":"content"},
         "metrics": {"rbo": named_partial(Metrics.rbo_score, p=0.999)}
     },

     "content_activity_disparity_gini_fork": {
         "question": 14,
         "scale": "population",
         "node_type":"content",
         'scenario1':True,
         'scenario2':False,
         'scenario3':True,
         "measurement": "getGiniCoef",
         "measurement_args":{"eventTypes":["ForkEvent"],"content_field":"content","nodeType":"content"},
         "metrics": {"absolute_difference": Metrics.absolute_difference}
     },
    
     "content_activity_disparity_palma_fork": {
         "question": 14,
         "scale": "population",
         "node_type":"content",
         'scenario1':True,
         'scenario2':False,
         'scenario3':True,
         "measurement": "getPalmaCoef",
         "measurement_args":{"eventTypes":["ForkEvent"],"content_field":"content","nodeType":"content"},
         "metrics": {"absolute_difference": Metrics.absolute_difference}
     },
    
      "content_activity_disparity_gini_push": {
          "question": 14,
          "scale": "population",
          "node_type":"content",
          'scenario1':True,
          'scenario2':False,
          'scenario3':True,
          "filters": {"event": ["PushEvent"]},
          "measurement": "getGiniCoef",
          "measurement_args":{"eventTypes":["PushEvent"],"content_field":"content","nodeType":"content"},
          "metrics": {"absolute_difference": Metrics.absolute_difference}
      },
    
      "content_activity_disparity_palma_push": {
          "question": 14,
          "scale": "population",
          "node_type":"content",
          'scenario1':True,
          'scenario2':False,
          'scenario3':True,
          "measurement": "getPalmaCoef",
          "measurement_args":{"eventTypes":["PushEvent"],"content_field":"content","nodeType":"content"},
          "metrics": {"absolute_difference": Metrics.absolute_difference}
      },
    
      "content_activity_disparity_gini_pullrequest": {
          "question": 14,
          "scale": "population",
          "node_type":"content",
          'scenario1':True,
          'scenario2':False,
          'scenario3':True,
          "measurement": "getGiniCoef",
          "measurement_args":{"eventTypes":["PullRequestEvent"],"content_field":"content","nodeType":"content"},
          "metrics": {"absolute_difference": Metrics.absolute_difference}
      },
    
      "content_activity_disparity_palma_pullrequest": {
          "question": 14,
          "scale": "population",
          "node_type":"content",
          'scenario1':True,
          'scenario2':False,
          'scenario3':True,
          "measurement_args":{"eventTypes":["PullRequestEvent"],"content_field":"content","nodeType":"content"},
          "measurement": "getPalmaCoef",
          "metrics": {"absolute_difference": Metrics.absolute_difference}
      },
    
      "content_activity_disparity_gini_issue": {
          "question": 14,
          "scale": "population",
          "node_type":"content",
          'scenario1':True,
          'scenario2':False,
          'scenario3':True,
          "measurement": "getGiniCoef",
          "measurement_args":{"eventTypes":["IssuesEvent"],"content_field":"content","nodeType":"content"},
          "metrics": {"absolute_difference": Metrics.absolute_difference}
      },
    
      "content_activity_disparity_palma_issue": {
          "question": 14,
          "scale": "population",
          "node_type":"content",
          'scenario1':True,
          'scenario2':False,
          'scenario3':True,
          "measurement": "getPalmaCoef",
          "measurement_args":{"eventTypes":["IssuesEvent"],"content_field":"content","nodeType":"content"},
          "metrics": {"absolute_difference": Metrics.absolute_difference}
      },
    
      "repo_trustingness":{
          "question": '15',
          "scale": "population",
          "node_type":"content",
          'scenario1':True,
          'scenario2':False,
          'scenario3':True,
          "filters":{"event":"PullRequestEvent"},
          "measurement":"getRepoPullRequestAcceptance",
          "metrics":{"ks_test":Metrics.ks_test}
          },

     "repo_issue_to_push":{
         "question":'31',
         "scale":"node",
         "node_type":"content",
         'scenario1':False,
         'scenario2':False,
         'scenario3':False,
         "measurement_args":{"eventTypes":contribution_events,"content_field":"content"},
         "measurement":"getEventTypeRatioTimeline",
         "metrics":{"rmse":Metrics.rmse}
         },

      "repo_event_counts_issue": {
         "question": '11',
         "scale": "population",
         "node_type": "content",
         'scenario1':True,
         'scenario2':False,
         'scenario3':True,
         "measurement": "getDistributionOfEventsByContent",
         "measurement_args":{"eventTypes":["IssuesEvent"],"content_field":"content"},
         "metrics":{"js_divergence":named_partial(Metrics.js_divergence, discrete=False),
                    "rmse":Metrics.rmse,
                    "r2": Metrics.r2}
         },
      "repo_event_counts_pull_request": {
         "question": '11',
         "scale": "population",
         "node_type": "content",
         'scenario1':True,
         'scenario2':False,
         'scenario3':True,
         "measurement": "getDistributionOfEventsByContent",
         "measurement_args":{"eventTypes":["PullRequestEvent"],"content_field":"content"},
         "metrics":{"js_divergence":named_partial(Metrics.js_divergence, discrete=False),
                    "rmse":Metrics.rmse,
                    "r2": Metrics.r2}
         },
      "repo_event_counts_push": {
         "question": '11',
         "scale": "population",
         "node_type": "content",
         'scenario1':True,
         'scenario2':False,
         'scenario3':True,
         "measurement": "getDistributionOfEventsByContent",
         "measurement_args":{"eventTypes":["PushEvent"],"content_field":"content"},
         "metrics":{"js_divergence":named_partial(Metrics.js_divergence, discrete=False),
                    "rmse":Metrics.rmse,
                    "r2": Metrics.r2}
         },
      "repo_user_continue_prop":{
         "question":"30",
         "scale":"node",
         'scenario1':False,
         'scenario2':False,
         'scenario3':False,
         "node_type":"content",
         "measurement":"propUserContinue",
         "measurement_args":{"eventTypes":contribution_events,"content_field":"content"},
         "metrics":{"rmse":Metrics.rmse}
         }
}

community_measurement_params = {
    #Community-level measurements
    "community_gini":{
        "question":'6',
        "scale":"community",
        "node_type":"content",
         'scenario1':True,
         'scenario2':False,
         'scenario3':True,
        "measurement":"getCommunityGini",
        "measurement_args":{"eventTypes":contribution_events,"community_field":"community","content_field":"content"},
        "metrics":{"absolute_difference": Metrics.absolute_difference}
        },

    "community_palma":{
        "question":'6',
        "scale":"community",
        "node_type":"content",
        'scenario1':True,
        'scenario2':False,
        'scenario3':True,
        "measurement":"getCommunityPalma",
        "measurement_args":{"eventTypes":contribution_events,"community_field":"community","content_field":"content"},
        "metrics":{"absolute_difference": Metrics.absolute_difference}
        },

    "community_geo_locations":{
        "question":'21',
        "scale":"community",
        "node_type":"user",
        'scenario1':True,
        'scenario2':False,
        'scenario3':True,
        "measurement":"userGeoLocation",
        "measurement_args":{"eventTypes":contribution_events,"community_field":"community"},
        "metrics":{"js_divergence":named_partial(Metrics.js_divergence, discrete=False)}
        },

    "community_event_proportions":{
        "question":'7',
        "scale":"community",
        "node_type":"content",
        'scenario1':True,
        'scenario2':False,
        'scenario3':True,
        "measurement":"getProportion",
         "measurement_args":{"eventTypes":contribution_events + popularity_events,"community_field":"community"},
        "metrics":{"js_divergence": named_partial(Metrics.js_divergence,discrete=True)}
        },
    "community_contributing_users":{
        "question":"20",
        "scale":"community",
        "node_type":"user",
        'scenario1':True,
        'scenario2':False,
        'scenario3':True,
        "measurement":"contributingUsers",
        "measurement_args":{"community_field":"community"},
        "metrics":{"absolute_difference":Metrics.absolute_difference}
        },
    "community_num_user_actions":{
        "question":"23",
        "scale":"community",
        "node_type":"user",
        'scenario1':True,
        'scenario2':False,
        'scenario3':True,
        "measurement":"getNumUserActions",
        "measurement_args":{"eventTypes":contribution_events,"community_field":"community","unit":"D"},
        "metrics":{"rmse": named_partial(Metrics.rmse, join="outer"),
                   "dtw": Metrics.dtw,
                   "js_divergence": named_partial(Metrics.js_divergence,discrete=False)
                   }
        },
    "community_burstiness":{
        "question":"9",
        "scale":"community",
        "node_type":"content",
        'scenario1':True,
        'scenario2':False,
        'scenario3':True,
        "measurement":"burstsInCommunityEvents",
        "measurement_args":{"eventTypes":contribution_events + popularity_events,"community_field":"community"},
        "metrics":{"absolute_difference":Metrics.absolute_difference}
        },
    "community_user_burstiness":{
        "question":"",
        "scale":"community",
        "node_type":"user",
        'scenario1':True,
        'scenario2':False,
        'scenario3':True,
        "measurement":"getUserBurstByCommunity",
        "measurement_args":{"community_field":"community"},
        "metrics":{'ks_test':Metrics.ks_test}
        },
    "community_issue_types":{
        "question":"8",
        "scale":"community",
        "node_type":"content",
        'scenario1':True,
        'scenario2':False,
        'scenario3':True,
        "measurement":"propIssueEvent",
        "metrics":{ "rmse": named_partial(Metrics.rmse,join='outer'),
                    "js_divergence": named_partial(Metrics.js_divergence, discrete=True)}
        },
    "community_user_account_ages":{
        "question":"10",
        "scale":"community",
        "node_type":"user",
        'scenario1':True,
        'scenario2':False,
        'scenario3':True,     
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
        'scenario1':False,
        'scenario2':False,
        'scenario3':False,
        "measurement":"computeTEUsers",
        "metrics": {"rbo": named_partial(Metrics.rbo_for_te, idx = 0,  wt = 0.9 ,ct = 30)}
        },
    
    "user_total":{
            "question":'18a2',
            "scale":"te",
            "node_type":"user",
            'scenario1':False,
            'scenario2':False,
            'scenario3':False, 
            "measurement":"computeTEUsers",
            "metrics": {"rbo": named_partial(Metrics.rbo_for_te, idx = 1, wt = 0.75 , ct = 10)}
            },    

    "user_event_interactions":{
            "question":'18b',
            "scale":"te",
            "node_type":"user",
            'scenario1':False,
            'scenario2':False,
            'scenario3':False,
            "measurement":"computeTEUserEvents",
            "metrics": {"rbo": named_partial(Metrics.rbo_for_te, idx = 0, wt = 0.9 ,ct = 25)}
            },  
    
    "repo_interactions":{
            "question":'18c1',
            "scale":"te",
            "node_type":"content",
            'scenario1':False,
            'scenario2':False,
            'scenario3':False,
            "measurement":"computeTERepos",
            "metrics": {"rbo": named_partial(Metrics.rbo_for_te, idx = 0, wt = 0.9, ct = 30)}
            }, 
    
    "repo_total":{
            "question":'18c2',
            "scale":"te",
            'scenario1':False,
            'scenario2':False,
            'scenario3':False,
            "node_type":"content",
            "measurement":"computeTERepos",
            "metrics": {"rbo": named_partial(Metrics.rbo_for_te, idx = 1, wt = 0.75, ct = 10)}
            }
    }


github_measurement_params = {}
github_measurement_params.update(user_measurement_params)
github_measurement_params.update(content_measurement_params)
github_measurement_params.update(community_measurement_params)
#measurement_params.update(te_measurement_params)
