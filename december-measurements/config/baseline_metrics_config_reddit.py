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


reddit_events = ["post","comment"]


user_measurement_params = {
    ### User Centric Measurements
     "user_unique_content": {
         'question': '17',
         "scale": "population",
         "node_type":"user",
         "scenario1":True,
         "scenario2":True,
         "scenario3":True,
         "measurement": "getUserUniqueContent",
         "measurement_args":{"eventTypes":reddit_events,"content_field":"root"},
         "metrics": {
             "js_divergence": named_partial(Metrics.js_divergence, discrete=False),
             "rmse": Metrics.rmse,
             "r2": Metrics.r2}
     },

     "user_activity_timeline": {
         "question": '19',
         "scale": "node",
         "node_type":"user",
         "scenario1":False,
         "scenario2":True,
         "scenario3":False,
         "measurement": "getUserActivityTimeline",
         "measurement_args":{"eventTypes":reddit_events},
         "metrics": {"rmse": Metrics.rmse,
                     "ks_test": Metrics.ks_test,
                     "dtw": Metrics.dtw}
    
     },

     "user_activity_distribution": {
         "question": '24a',
         "scale": "population",
         "node_type":"user",
         "scenario1":True,
         "scenario2":True,
         "scenario3":True,
         "measurement": "getUserActivityDistribution",
         "measurement_args":{"eventTypes":reddit_events},
         "metrics": {"rmse": Metrics.rmse,
                     "r2": Metrics.r2,
                     "js_divergence": named_partial(Metrics.js_divergence, discrete=True)}
     },
    
     "most_active_users": {
         "question": '24b',
         "scale": "population",
         "node_type":"user",
         "scenario1":True,
         "scenario2":True,
         "scenario3":True,
         "measurement": "getMostActiveUsers",
         "measurement_args":{"eventTypes":reddit_events},
         "metrics": {"rbo": named_partial(Metrics.rbo_score, p=0.999)}
     },

     "user_popularity": {
         "question": '25',
         "scale": "population",
         "node_type":"user",
         "scenario1":True,
         "scenario2":True,
         "scenario3":True,
         "measurement": "getUserPopularity",
         "measurement_args":{"eventTypes":reddit_events,"content_field":"root"},
         "metrics": {"rbo": named_partial(Metrics.rbo_score, p=0.999)}
     },

     "user_gini_coef": {
         "question": '26a',
         "scale": "population",
         "node_type":"user",
         "scenario1":True,
         "scenario2":True,
         "scenario3":True,
         "measurement": "getGiniCoef",
         "measurement_args":{"nodeType":"user","eventTypes":reddit_events},
         "metrics": {"absolute_difference": Metrics.absolute_difference,
                     "absolute_percentage_error": Metrics.absolute_percentage_error}
     },

     "user_palma_coef": {
         "question": '26b',
         "scale": "population",
         "node_type":"user",
         "scenario1":True,
         "scenario2":True,
         "scenario3":True,
         "measurement": "getPalmaCoef",
         "measurement_args":{"nodeType":"user","eventTypes":reddit_events},
         "metrics": {"absolute_percentage_error": Metrics.absolute_percentage_error}
     },

     "user_diffusion_delay": {
         "question": '27',
         "scale": "population",
         "node_type":"user",
         "scenario1":True,
         "scenario2":True,
         "scenario3":True,
         "measurement": "getUserDiffusionDelay",
         "measurement_args":{"eventTypes":reddit_events},
         "metrics": {"ks_test": Metrics.ks_test}
     }

}

content_measurement_params = {
    ##Content-centric measurements
     "content_diffusion_delay": {
         "question": 1,
         "scale": "node",
         "node_type":"content",
         "scenario1":False,
         "scenario2":True,
         "scenario3":False,
         "measurement": "getContentDiffusionDelay",
         "measurement_args":{"eventTypes":["comment"],"time_bin":"h"},
         "metrics": {"ks_test": Metrics.ks_test,
                     "js_divergence": named_partial(Metrics.js_divergence, discrete=False)},
     },
   
     "content_growth": {
         "question": 2,
         "scale": "node",
         "node_type":"content",
         "scenario1":False,
         "scenario2":True,
         "scenario3":False,
         "measurement": "getContentGrowth",
         "measurement_args":{"eventTypes":reddit_events,"time_bin":"h"},
         "metrics": {"rmse": named_partial(Metrics.rmse, join="outer"),
                     "dtw": Metrics.dtw}
     },
   
     "content_contributors": {
         "question": 4,
         "scale": "node",
         "node_type":"content",
         "scenario1":False,
         "scenario2":True,
         "scenario3":False,
         "measurement": "getContributions",
         "measurement_args":{"eventTypes":reddit_events},
         "metrics": {"rmse": named_partial(Metrics.rmse, join="outer"),
                     "dtw": Metrics.dtw}
     },
      
    "content_event_distribution_dayofweek": {
        "question": 5,
        "scale": "node",
        "node_type":"content",
        "scenario1":False,
        "scenario2":True,
        "scenario3":False,
        "measurement": "getDistributionOfEvents",
        "measurement_args":{"weekday":True},
        "metrics": {"js_divergence": named_partial(Metrics.js_divergence, discrete=True)}
    },
    
     "content_liveliness_distribution": {
         "question": 13,
         "scale": "population",
         "node_type":"content",
         "scenario1":True,
         "scenario2":True,
         "scenario3":True,
         "measurement": "getDistributionOfEventsByContent",
         "measurement_args":{"eventTypes":["comment"],"content_field":"root"},
         "metrics": {"js_divergence": named_partial(Metrics.js_divergence, discrete=False),
                     "rmse": Metrics.rmse,
                     "r2": Metrics.r2}
     },
    
     "content_liveliness_topk": {
         "question": 13,
         "scale": "population",
         "node_type":"content",
         "scenario1":True,
         "scenario2":True,
         "scenario3":True,
         "measurement": "getTopKContent",
         "measurement_args":{"k":5000,"eventTypes":["comment"],"content_field":"root"},
         "metrics": {"rbo": named_partial(Metrics.rbo_score, p=0.999)}
     },

      "content_activity_disparity_gini_comment": {
          "question": 14,
          "scale": "population",
          "node_type":"repo",
          "scenario1":True,
          "scenario2":True,
          "scenario3":True,
          "filters": {"event": ["comment"]},
          "measurement": "getGiniCoef",
          "measurement_args":{"eventTypes":["comment"]},
          "metrics": {"absolute_difference": Metrics.absolute_difference,
                      "absolute_percentage_error": Metrics.absolute_percentage_error}
      },
    
      "content_activity_disparity_palma_comment": {
          "question": 14,
          "scale": "population",
          "node_type":"repo",
          "scenario1":True,
          "scenario2":True,
          "scenario3":True,
          "measurement": "getPalmaCoef",
          "measurement_args":{"eventTypes":["comment"]},
          "metrics": {"absolute_percentage_error": Metrics.absolute_percentage_error}
      },
    
      "subreddit_user_continue_prop":{
         "question":"30",
         "scale":"node",
         "node_type":"repo",
         "scenario1":True,
         "scenario2":True,
         "scenario3":True,
         "measurement":"propUserContinue",
         "measurement_args":{"eventTypes":["comment"],"content_field":"subreddit"},
         "metrics":{"rmse":Metrics.rmse}
         },
     "subreddit_post_to_comment":{
         "question":'31',
         "scale":"node",
         "node_type":"content",
         "scenario1":True,
         "scenario2":True,
         "scenario3":True,
         "measurement_args":{"eventTypes":reddit_events,"event1":"post","event2":"comment","content_field":"subreddit"},
         "measurement":"getEventTypeRatioTimeline",
         "metrics":{"rmse":Metrics.rmse}
         }
}

community_measurement_params = {
    #Community-level measurements
    "community_gini":{
        "question":'6',
        "scale":"community",
        "node_type":"content",
        "scenario1":True,
        "scenario2":True,
        "scenario3":True,
        "measurement":"getCommunityGini",
        "measurement_args":{"eventTypes":reddit_events,"community_field":"subreddit","content_field":"root"},
        "metrics":{"absolute_difference": Metrics.absolute_difference,
                   "absolute_percentage_error": Metrics.absolute_percentage_error}
        },

    "community_palma":{
        "question":'6',
        "scale":"community",
        "node_type":"content",
        "scenario1":True,
        "scenario2":True,
        "scenario3":True,
        "measurement":"getCommunityPalma",
        "measurement_args":{"eventTypes":reddit_events,"community_field":"subreddit","content_field":"root"},
        "metrics":{"absolute_percentage_error": Metrics.absolute_percentage_error}
        },

    "community_geo_locations":{
        "question":'21',
        "scale":"community",
        "node_type":"user",
        "scenario1":True,
        "scenario2":True,
        "scenario3":True,
        "measurement":"userGeoLocation",
        "measurement_args":{"eventTypes":reddit_events,"community_field":"subreddit"},
        "metrics":{"js_divergence":named_partial(Metrics.js_divergence, discrete=False)}
        },

    "community_event_proportions":{
        "question":'7',
        "scale":"community",
        "node_type":"content",
        "scenario1":True,
        "scenario2":True,
        "scenario3":True,
        "measurement":"getProportion",
         "measurement_args":{"eventTypes":reddit_events,"community_field":"subreddit"},
        "metrics":{"js_divergence": named_partial(Metrics.js_divergence,discrete=True)}
        },
    "community_contributing_users":{
        "question":"20",
        "scale":"community",
        "node_type":"user",
        "scenario1":True,
        "scenario2":True,
        "scenario3":True,
        "measurement":"contributingUsers",
        "measurement_args":{"eventTypes":reddit_events,"community_field":"subreddit"},
        "metrics":{"absolute_difference":Metrics.absolute_difference,
                   "absolute_percentage_error": Metrics.absolute_percentage_error}
        },
    "community_num_user_actions":{
        "question":"23",
        "scale":"community",
        "node_type":"user",
        "scenario1":True,
        "scenario2":True,
        "scenario3":True,
        "measurement":"getNumUserActions",
        "measurement_args":{"eventTypes":reddit_events,"unit":'D',"community_field":"subreddit"},
        "metrics":{"rmse": named_partial(Metrics.rmse, join="outer"),
                   "dtw": Metrics.dtw,
                   "js_divergence": named_partial(Metrics.js_divergence,discrete=False)
                   }
        },
    "community_burstiness":{
        "question":"9",
        "scale":"community",
        "node_type":"content",
        "scenario1":True,
        "scenario2":True,
        "scenario3":True,
        "measurement":"burstsInCommunityEvents",
        "measurement_args":{"eventTypes":reddit_events,"community_field":"subreddit"},
        "metrics":{"absolute_difference":Metrics.absolute_difference,
                   "absolute_percentage_error": Metrics.absolute_percentage_error}
        },

    "community_user_burstiness":{
        "question":"",
        "scale":"community",
        "node_type":"user",
        "scenario1":True,
        "scenario2":True,
        "scenario3":True,
        "measurement":"getUserBurstByCommunity",
        "metrics":{'ks_test':Metrics.ks_test}
        },

    "community_user_account_ages":{
        "question":"10",
        "scale":"community",
        "node_type":"user",
        "scenario1":True,
        "scenario2":True,
        "scenario3":True,
        "measurement":"ageOfAccounts",
        "metrics":{'ks_test':Metrics.ks_test}
        }
    }

reddit_measurement_params = {}

reddit_measurement_params.update(user_measurement_params)
reddit_measurement_params.update(content_measurement_params)
reddit_measurement_params.update(community_measurement_params)
