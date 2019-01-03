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


twitter_events = ["tweet","retweet","quote","reply"]


user_measurement_params = {
    ### User Centric Measurements
     "user_unique_content": {
         'question': '17',
         "scale": "population",
         "node_type":"user",
         'scenario1':True,
         'scenario2':True,
         'scenario3':True,
         "measurement": "getUserUniqueContent",
         "measurement_args":{"eventTypes":twitter_events,"content_field":"root"},
         "metrics": {
             "js_divergence": named_partial(Metrics.js_divergence, discrete=False),
             "rmse": Metrics.rmse,
             "nrmse": named_partial(Metrics.rmse,relative=True),
             "r2": Metrics.r2}
     },

#     "user_activity_timeline": {
#         "question": '19',
#         "scale": "node",
#         "node_type":"user",
#         'scenario1':False,
#         'scenario2':True,
#         'scenario3':False,
#         "measurement": "getUserActivityTimeline",
#         "measurement_args":{"eventTypes":twitter_events},
#         "metrics": {"rmse": Metrics.rmse,
#                     "nrmse": named_partial(Metrics.rmse,relative=True),
#                     "ks_test": Metrics.ks_test,
#                     "dtw": Metrics.dtw}    
#     },

     "user_activity_distribution": {
         "question": '24a',
         "scale": "population",
         "node_type":"user",
         'scenario1':True,
         'scenario2':True,
         'scenario3':True,
         "measurement": "getUserActivityDistribution",
         "measurement_args":{"eventTypes":twitter_events},
         "metrics": {"rmse": Metrics.rmse,
                     "nrmse": named_partial(Metrics.rmse,relative=True),
                     "r2": Metrics.r2,
                     "js_divergence": named_partial(Metrics.js_divergence, discrete=True)}
     },
    
     "most_active_users": {
         "question": '24b',
         "scale": "population",
         "node_type":"user",
         'scenario1':True,
         'scenario2':True,
         'scenario3':True,
         "measurement": "getMostActiveUsers",
         "measurement_args":{"k":30,"eventTypes":twitter_events},
         "metrics": {"rbo": named_partial(Metrics.rbo_score, p=0.84)}
     },

     "user_popularity": {
         "question": '25',
         "scale": "population",
         "node_type":"user",
         'scenario1':True,
         'scenario2':True,
         'scenario3':True,
         "measurement": "getUserPopularity",
         "measurement_args":{"k":30,"eventTypes":twitter_events,"content_field":"root"},
         "metrics": {"rbo": named_partial(Metrics.rbo_score, p=0.84)}
     },

     "user_gini_coef": {
         "question": '26a',
         "scale": "population",
         "node_type":"user",
         'scenario1':True,
         'scenario2':True,
         'scenario3':True,
         "measurement": "getGiniCoef",
         "measurement_args":{"nodeType":"user","eventTypes":twitter_events},
         "metrics": {"absolute_difference": Metrics.absolute_difference,
                     "absolute_percentage_error":Metrics.absolute_percentage_error}
     },

     "user_palma_coef": {
         "question": '26b',
         "scale": "population",
         "node_type":"user",
         'scenario1':True,
         'scenario2':True,
         'scenario3':True,
         "measurement": "getPalmaCoef",
         "measurement_args":{"nodeType":"user","eventTypes":twitter_events},
         "metrics": {"absolute_percentage_error":Metrics.absolute_percentage_error,
                     "absolute_difference":Metrics.absolute_difference}
     },

     #"user_diffusion_delay": {
     #    "question": '27',
     #    "scale": "population",
     #    "node_type":"user",
     #    'scenario1':True,
     #    'scenario2':True,
     #    'scenario3':True,
     #    "measurement": "getUserDiffusionDelay",
     #    "measurement_args":{"eventTypes":twitter_events},
     #    "metrics": {"ks_test": Metrics.ks_test}
     #}

}

content_measurement_params = {
    ##Content-centric measurements
#     "content_diffusion_delay": {
#         "question": 1,
#         "scale": "node",
#         "node_type":"content",
#         "scenario1":False,
#         "scenario2":True,
#         "scenario3":False,
#         "measurement": "getContentDiffusionDelay",
#         "measurement_args":{"eventTypes":["reply",'retweet','quote'],"time_bin":"h","content_field":"root"},
#         "metrics": {"ks_test": Metrics.ks_test,
#                     "js_divergence": named_partial(Metrics.js_divergence, discrete=False)},
#     },
   
#     "content_growth": {
#         "question": 2,
#         "scale": "node",
#         "node_type":"content",
#         "scenario1":False,
#         "scenario2":True,
#         "scenario3":False,
#         "measurement": "getContentGrowth",
#         "measurement_args":{"eventTypes":twitter_events,"time_bin":"d","content_field":"root"},
#         "metrics": {"rmse": named_partial(Metrics.rmse, join="outer"),
#                     "dtw": Metrics.dtw}
#     },
   
#     "content_contributors": {
#         "question": 4,
#         "scale": "node",
#         "node_type":"content",
#         "scenario1":False,
#         "scenario2":True,
#         "scenario3":False,
#         "measurement": "getContributions",
#         "measurement_args":{"eventTypes":twitter_events,"content_field":"root"},
#         "metrics": {"rmse": named_partial(Metrics.rmse, join="outer"),
#                     "dtw": Metrics.dtw}
#     },
      
#    "content_event_distribution_dayofweek": {
#        "question": 5,
#        "scale": "node",
#        "node_type":"content",
#         "scenario1":False,
#         "scenario2":True,
#         "scenario3":False,
#        "measurement": "getDistributionOfEvents",
#        "measurement_args":{"weekday":True,"content_field":"root"},
#        "metrics": {"js_divergence": named_partial(Metrics.js_divergence, discrete=True)}
#    },
    
     "content_liveliness_distribution": {
         "question": 13,
         "scale": "population",
         "node_type":"content",
         "scenario1":True,
         "scenario2":True,
         "scenario3":True,
         "measurement": "getDistributionOfEventsByContent",
         "measurement_args":{"eventTypes":["reply"],"content_field":"root"},
         "metrics": {"js_divergence": named_partial(Metrics.js_divergence, discrete=False)}
     },
    
#     "content_liveliness_topk": {
#         "question": 13,
#         "scale": "population",
#         "node_type":"content",
#         "scenario1":False,
#         "scenario2":True,
#         "scenario3":False,
#         "measurement": "getTopKContent",
##         "measurement_args":{"k":50,"eventTypes":["reply"],"content_field":"root"},
#         "metrics": {"rbo": named_partial(Metrics.rbo_score, p=0.9)}
#     },

     "content_popularity_distribution": {
         "question": 13,
         "scale": "population",
         "node_type":"content",
         "scenario1":False,
         "scenario2":True,
         "scenario3":False,
         "measurement": "getDistributionOfEventsByContent",
         "measurement_args":{"eventTypes":["retweet"],"content_field":"root"},
         "metrics": {"js_divergence": named_partial(Metrics.js_divergence, discrete=False)}
     },
    
#     "content_popularity_topk": {
#         "question": 13,
#         "scale": "population",
#         "node_type":"content",
#         "scenario1":True,
#         "scenario2":True,
#         "scenario3":True,
#         "measurement": "getTopKContent",
#         "measurement_args":{"k":5000,"eventTypes":["retweet"],"content_field":"root"},
#         "metrics": {"rbo": named_partial(Metrics.rbo_score, p=0.999)}
#     },

      "content_activity_disparity_gini_retweet": {
          "question": 14,
          "scale": "population",
          "node_type":"content",
         "scenario1":True,
         "scenario2":True,
         "scenario3":True,
          "measurement": "getGiniCoef",
          "measurement_args":{"eventTypes":["retweet"],"nodeType":"root"},
          "metrics": {"absolute_difference": Metrics.absolute_difference,
                      "absolute_percentage_error":Metrics.absolute_percentage_error}
      },
    
      "content_activity_disparity_palma_retweet": {
          "question": 14,
          "scale": "population",
          "node_type":"content",
         "scenario1":True,
         "scenario2":True,
         "scenario3":True,
          "measurement": "getPalmaCoef",
          "measurement_args":{"eventTypes":["retweet"],"nodeType":"root"},
          "metrics": {"absolute_percentage_error":Metrics.absolute_percentage_error,
                     "absolute_difference":Metrics.absolute_difference}
      },
#      "content_activity_disparity_gini_quote": {
#          "question": 14,
#          "scale": "population",
#          "node_type":"content",
#          "scenario1":True,
#          "scenario2":True,
#          "scenario3":True,
#          "measurement": "getGiniCoef",
#          "measurement_args":{"eventTypes":["quote"],"nodeType":"root"},
#          "metrics": {"absolute_difference": Metrics.absolute_difference,
#                      "absolute_percentage_error":Metrics.absolute_percentage_error}
#      },
    
#      "content_activity_disparity_palma_quote": {
#          "question": 14,
#          "scale": "population",
#          "node_type":"content",
#          "scenario1":True,
#          "scenario2":True,
#          "scenario3":True,
#          "measurement": "getPalmaCoef",
#          "measurement_args":{"eventTypes":["quote"],"nodeType":"root"},
#          "metrics": {"absolute_percentage_error":Metrics.absolute_percentage_error,
#                      "absolute_difference":Metrics.absolute_difference}
#      },
#      "content_activity_disparity_gini_reply": {
#          "question": 14,
#          "scale": "population",
#          "node_type":"content",
#          "scenario1":True,
#          "scenario2":True,
#          "scenario3":True,
#          "measurement": "getGiniCoef",
#          "measurement_args":{"eventTypes":["reply"],"nodeType":"root"},
#          "metrics": {"absolute_difference": Metrics.absolute_difference,
#                      "absolute_percentage_error":Metrics.absolute_percentage_error}
#      },
    
#      "content_activity_disparity_palma_reply": {
#          "question": 14,
#          "scale": "population",
#          "node_type":"content",
#          "scenario1":True,
#          "scenario2":True,
#          "scenario3":True,
#          "measurement": "getPalmaCoef",
#          "measurement_args":{"eventTypes":["reply"],"nodeType":"root"},
#          "metrics": {"absolute_percentage_error":Metrics.absolute_percentage_error,
#                      "absolute_difference":Metrics.absolute_difference}
#      }


}


twitter_scenario1_measurement_params_cve = {}
twitter_scenario1_measurement_params_cve.update(user_measurement_params)
twitter_scenario1_measurement_params_cve.update(content_measurement_params)
