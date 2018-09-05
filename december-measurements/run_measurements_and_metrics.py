import pandas as pd
from functools import partial, update_wrapper

import Metrics
import ContentCentricMeasurements
import UserCentricMeasurements

from BaselineMeasurements import *
from network_measurements import *
from cascade_measurements import *

import pprint

import sys
sys.path.append('config')
from baseline_metrics_config_reddit import *
from baseline_metrics_config_twitter import *
from baseline_metrics_config_github import *
from cascade_metrics_config import *
from network_metrics_config import *

sys.path.append('plotting')
import charts
from visualization_config import measurement_plot_params
import os
import tqdm
import transformer
import traceback


def named_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    partial_func.varnames = func.__code__.co_varnames
    return partial_func


def run_metrics(ground_truth, simulation, measurement_name, measurement_params,
                plot_flag=False,show=False,plot_dir=''):

    """
    Run all of the assigned metrics for a given measurement.

    Inputs:
    ground_truth - Measurements object of ground truth data
    simulation - Measurements object of simulated data
    measurement_name - Name of measurement corresponding to keys of measurement_params
    plot_flag - Boolean indicating whether to generate plots (True) or not (False)
    show - Boolean indicating whether to display plots to screen
    plot_dir - Directory to save plots.  If empty string, the plots will not be saved.


    Outputs:
    measurement_on_gt - Output of the measurement for the ground truth data
    measurement_on_sim - Output of the measurement for the simulation data
    metrics_output - Dictionary containing metric results for each metric assigned to the measurement
    """


    #ground_truth measurement
    if not isinstance(ground_truth,dict):
        measurement_on_gt = run_measurement(ground_truth,measurement_name,measurement_params,plot_flag=False,simulation=False)
    else:
        measurement_on_gt = ground_truth[measurement_name]

    print('measurement_name',measurement_name)
    measurement_on_sim = run_measurement(simulation,measurement_name,measurement_params,plot_flag=False,simulation=True)
    
    if plot_flag:
        generate_plot(simulation = measurement_on_sim, ground_truth = measurement_on_gt, 
                      measurement_name=measurement_name, show=show, plot_dir=plot_dir)

    metrics_output = {}

    p = measurement_params[measurement_name]
    metrics = p['metrics']

    if p["scale"] in ["node","community"]:
        
        if measurement_on_gt is None:
            measurement_on_gt = {}

        for node in measurement_on_gt:
            metrics_output[node] = {}


    #iterate over the metrics assigned to the measurement
    for m, metric_function in metrics.items():
        print("Calculating {} for {}".format(metric_function.__name__, p['measurement']))

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


def run_measurement(data, measurement_name, measurement_params, plot_flag=False,simulation=True,
                    show=False,plot_dir=''):

    """
    Run all of the assigned metrics for a single given measurement.

    Inputs:
    data - Measurements object of the ground truth or simulation data for which the measurement will be calculated
    measurement_name - Name of measurement corresponding to keys of measurement_params
    measurement_params - A measurement parameters dictionary indicating the measurements and measurement properties of interest.  These can be found in the metrics_config files. 
    simulation - Boolean indicating whether the data comes from a simulation or ground truth data file.  Used for the plotting code.
    plot_flag - Boolean indicating whether to generate plots (True) or not (False)
    show - Boolean indicating whether to display plots to screen
    plot_dir - Directory to save plots.  If empty string, the plots will not be saved.


    Outputs:
    measurement - The output of the specified measurement function on the input data
    """


    p = measurement_params[measurement_name]
    if "measurement_args" in p:
        measurement_args = p["measurement_args"]
    else:
        measurement_args = {}

    if hasattr(data,p['measurement']):
        measurement_function = getattr(data,p['measurement'])
        if simulation:
            print("Measuring {} for simulation data".format(measurement_function.__name__))
        else:
            print("Measuring {} for ground truth data".format(measurement_function.__name__))
        measurement = measurement_function(**measurement_args)
        
        if plot_flag:
            if simulation:
                generate_plot(simulation=measurement,measurement_name=measurement_name,
                              show=show,plot_dir=plot_dir)
            else:
                generate_plot(ground_truth=measurement,measurement_name=measurement_name,
                              show=show, plot_dir=plot_dir)

        return measurement
  

def generate_plot(simulation=None, ground_truth = None, measurement_name='',show=False,plot_dir=''):

    """
    Generates a visualiztion of a given measurement and either displays to screen or saves the plot

    Inputs:
    simulation - Output of measurement function on the simulation data
    ground_truth - Output of measurement function on the ground truth data
    measurement_name - The name of the measurement being plotted
    show - Boolean indicating whether to display plots to screen
    plot_dir - Directory to save plots.  If empty string, the plots will not be saved.
    """

    if measurement_name in measurement_plot_params and not (simulation is None and ground_truth is None) and \
            not ((not simulation is None and len(simulation) == 0) and (not ground_truth is None and len(ground_truth) == 0)):

         #get plotting parameters for given measurement
         params = measurement_plot_params[measurement_name]
         
         #keys are the IDs for nodes and communities to extract individual measurements from the dictionary
         keys = ['']
         if 'plot_keys' in params:
             try:
                 keys = simulation.keys()    
             except:
                 keys = ground_truth.keys()

         #only generate limited number of plots for specific nodes/communities if showing to screen
         if show and len(keys) > 3:
             keys = keys[:3]
         
         #loop over individual nodes or communities
         for key in keys:

             #preprocess measurement output to prepare for plotting
             if key != '':
                 df = transformer.to_DataFrame(params['data_type'])(sim_data=simulation,ground_truth_data=ground_truth,key=key)
             else:
                 df = transformer.to_DataFrame(params['data_type'])(sim_data=simulation,ground_truth_data=ground_truth)

             if not df is None:
                 for p in params['plot']:
                     #generate plot
                     fig = charts.chart_factory(p)(df,params['x_axis'],params['y_axis'],
                                                   (key + ' ' + measurement_name).lstrip(),**params)

                     if plot_dir != '' and not fig is None:
                         if key != '':
                             charts.save_charts(fig,plot_dir + '/' + measurement_name + '_' + key.replace('/','@') + '.png')
                         else:
                             charts.save_charts(fig,plot_dir + '/' + measurement_name + '.png')

                     if show and not fig is None:
                         charts.show_charts()


def check_measurement(m_info,filters):

    """
    Determine whether a given measurement should be included based on the filters.

    Inputs:
    m_info - A dictionary containing the configuration parameters for an individual measurement.
    filters - A dictionary containing a set of configuration parameter values that should be included

    Output:
    include - Boolean indicating whether to include the given measurement

    """


    include = True
    for filter_field,filter_values in filters.iteritems():
        try:
            iter(filter_values)
        except:
            filter_values = [filter_values]

        if not m_info[filter_field] in filter_values:
            include = False

    return include


def run_all_measurements(data, measurement_params, filters = {},output_dir = None, plot_flag=False,
                         show=False,plot_dir=''):

    """
    Calculate metrics for multiple measurements.

    Inputs:
    data - Measurements object with with ground truth or simulation data.
    measurement_params - A measurement parameters dictionary indicating the measurements and measurement properties of interest.  These can be found in the metrics_config files.
    filters - Dictionary indicating subsets of the measurements that should be run.  The dictionary keys can be "node","scale", and "scenario".  For example, {"node_type":["population","community"],"scenario1":True} would run all population and community-level measurements that are included in scenario 1.
    output_dir - Directory to save output pickle files containing the mesurements outputs.  If empty string, the pickle files will not be saved.
    plot_flag - Boolean indicating whether to generate plots (True) or not (False)
    show - Boolean indicating whether to display plots to screen
    plot_dir - Directory to save plots.  If empty string, the plots will not be saved.


    Output:
    results - A dictionary with the measurement_names as keys and measurement calculation results as values

    """

    #select measurements of desired scale and node type
    measurements = [m for m, m_info in measurement_params.items() if check_measurement(m_info,filters)]

    results = {}
    for measurement_name in measurements:
        meas = run_measurement(data, measurement_name,measurement_params,plot_flag=plot_flag,
                               show=show,plot_dir=plot_dir)
        results[measurement_name] = meas
        if not output_dir is None:
            with open(output_dir + '/' + measurement_name + '.pkl','w') as f:
                pickle.dump(meas,f)
                
    return results


def load_measurements(directory, measurement_params, filters={}):

    """
    Load in saved measurement pickle files.

    Inputs:
    directory - A directory which contains the stored measurement pickle files.  These pickle files can be saved by using the output_dir argument in the run_all_measurements function.
    measurement_params - A measurement parameters dictionary indicating the measurements and measurement properties of interest.  These can be found in the metrics_config files.
    filters - Dictionary indicating subsets of the measurements that should be run.  The dictionary keys can be "node","scale", and "scenario".  For example, {"node_type":["population","community"],"scenario1":True} would run all population and community-level measurements that are included in scenario 1.

    Output:
    results - A dictionary with the measurement_names as keys and measurement calculation results as values

    """


    #select measurements of desired scale and node type
    measurements = [m for m, m_info in measurement_params.items() if check_measurement(m_info,filters)]

    results = {}
    for measurement_name in measurements:
        print('loading ' + directory + '/' + measurement_name + '.pkl...')
        with open(directory + '/' + measurement_name + '.pkl','r') as f:
            try:
                meas = pickle.load(f)
                results[measurement_name] = meas
            except:
                ''

    return results


def run_all_metrics(ground_truth, simulation, measurement_params, filters = {}, plot_flag=False,
                    show=False,plot_dir=''):

    """
    Calculate metrics for multiple measurements.

    Inputs:
    ground_truth - Measurements object with ground truth data or a ground truth measurement output dictionary.  If it is a Measurements object the measurements will be calculated for the groud truth.  If it is a dictionary, the pre-calculated measurements will be used. 
    simulation - Simulation Measurements object
    filters - Dictionary indicating subsets of the measurements that should be run.  The dictionary keys can be "node","scale", and "scenario".  For example, {"node_type":["population","community"],"scenario1":True} would run all population and community-level measurements that are included in scenario 1.
    plot_flag - Boolean indicating whether to generate plots (True) or not (False)
    show - Boolean indicating whether to display plots to screen
    plot_dir - Directory to save plots.  If empty string, the plots will not be saved.

    Outputs:
    results - A dictionary containing the metrics outputs for each measurement.  The measurement names are the keys and the a dictionary containing each metric is the value.
    """

    results = {}
    #select measurements of desired scale and node type
    measurements = [m for m, m_info in measurement_params.items() if check_measurement(m_info,filters)]

    for measurement_name in measurements:
        gt, sim, metric_results = run_metrics(ground_truth, simulation, measurement_name,measurement_params,plot_flag=plot_flag,
                                              show=show,plot_dir=plot_dir)
        results[measurement_name] = metric_results
    return results


def network_examples(platform="github"):

    ground_truth = pd.read_csv(platform + '_data_sample.csv')
    simulation = ground_truth.copy()

    measurement_class = {"github":GithubNetworkMeasurements,
                         "twitter":TwitterNetworkMeasurements,
                         "reddit":RedditNetworkMeasurements}

    node_ids = {"github":"nodeID",
                "twitter":"rootID",
                "reddit":"rootID"}

    #instantiate Measurement objects for both the ground truth and simulation data
    #pass test=True to run on small data subset, test=False to run on full data set
    gt_measurements = measurement_class[platform](data=ground_truth,node1='nodeUserID',node2=node_ids[platform],test=True)
    sim_measurements = measurement_class[platform](data=simulation,node1='nodeUserID',node2=node_ids[platform],test=True)

    #run individual measurement
    meas = run_measurement(sim_measurements, "number_of_nodes", 
                           network_measurement_params)


    #run individual metric
    gt_measurement, sim_measurement, metric = run_metrics(gt_measurements, sim_measurements, "number_of_nodes", 
                                                          network_measurement_params)
    pprint.pprint(metric)


    #run all assigned measurements
    #the measurement outputs will be saved as pickle files in the measurements_output directory
    meas = run_all_measurements(gt_measurements,network_measurement_params,output_dir='measurements_output/')


    #run all assigned metrics
    metrics = run_all_metrics(gt_measurements,sim_measurements,network_measurement_params)
    pprint.pprint(metrics)

    #load ground truth measurements from saved pickle files 
    gt_measurement = load_measurements('measurements_output/',network_measurement_params)

    #run all assigned metrics using the loaded measurements (without recalculating the measurements on the ground truth)
    metrics = run_all_metrics(gt_measurements,sim_measurements,network_measurement_params)
    pprint.pprint(metrics)
    

def cascade_examples(platform):

    ground_truth = pd.read_csv(platform + '_data_sample.csv',nrows=1000)
    simulation = ground_truth.copy()


    #instantiate Measurement objects for both the ground truth and simulation data
    #cascade measurements currently need a seperate objects for node level and for population and community level
    #these will be combined in a future release
    #the input file the node level measurements (SingleCascadeMeasurement) should contain the data for an individual cascade only
    #the input file for the community and population level (CascadeCollectionMeasurements) should contain all of the data
    gt_measurements_node = SingleCascadeMeasurements(ground_truth)
    sim_measurements_node = SingleCascadeMeasurements(simulation)

    gt_measurements_pop_community = CascadeCollectionMeasurements(ground_truth)
    sim_measurements_pop_community = CascadeCollectionMeasurements(simulation)


    #run individual measurement
    meas = run_measurement(sim_measurements_node, "cascade_size_over_time", 
                           cascade_measurement_params)
    meas = run_measurement(sim_measurements_pop_community,"population_cascade_size_distribution", 
                           cascade_measurement_params)


    #run individual metric
    gt_measurement, sim_measurement, metric = run_metrics(gt_measurements_node, sim_measurements_node, "cascade_size_over_time", 
                                                          cascade_measurement_params)
    pprint.pprint(metric)
    gt_measurement, sim_measurement, metric = run_metrics(gt_measurements_pop_community, sim_measurements_pop_community, "population_cascade_size_distribution", 
                                                          cascade_measurement_params)
    pprint.pprint(metric)


    #run all assigned measurements
    #the measurement outputs will be saved as pickle files in the measurements_output directory
    meas = run_all_measurements(gt_measurements_node,cascade_measurement_params,output_dir='measurements_output/')
    meas = run_all_measurements(gt_measurements_pop_community,cascade_measurement_params,output_dir='measurements_output/')


    #run all assigned metrics
    metrics = run_all_metrics(gt_measurements_node,sim_measurements_node,cascade_measurement_params)
    pprint.pprint(metrics)
    metrics = run_all_metrics(gt_measurements_pop_community,sim_measurements_pop_community,cascade_measurement_params)
    pprint.pprint(metrics)

    

def baseline_examples(platform="github"):


    ground_truth = pd.read_csv(platform + '_data_sample.csv')
    simulation = ground_truth.copy()

    configs = {"github":github_measurement_params,
               "twitter":twitter_measurement_params,
               "reddit":reddit_measurement_params}

    #specify node IDs for node-level measurements here
    content_ids = []
    user_ids = []
    
    #instantiate Measurement objects for both the ground truth and simulation data
    gt_measurements = BaselineMeasurements(ground_truth,platform=platform,content_node_ids=content_ids,user_node_ids = user_ids)
    sim_measurements = BaselineMeasurements(simulation,platform=platform,content_node_ids=content_ids, user_node_ids = user_ids)


    #run individual measurement
    #the plots will be displayed to screen
    meas = run_measurement(sim_measurements, "user_unique_content", 
                           configs[platform],show=True,plot_dir='',plot_flag=True)


    #run individual metric
    #the plots will be displayed to screen
    gt_measurement, sim_measurement, metric = run_metrics(gt_measurements, sim_measurements, "user_unique_content", 
                                                          configs[platform],show=True,plot_dir='',plot_flag=True)
    pprint.pprint(metric)


    #run all assigned measurements
    #the plots will be saved in the plots directory
    #the measurement outputs will be saved as pickle files in the measurements_output directory
    meas = run_all_measurements(gt_measurements,configs[platform],show=False,plot_dir='plots/',
                                plot_flag=True,output_dir='measurements_output/')



    #run all assigned metrics
    #no plots will be generated
    metrics = run_all_metrics(gt_measurements,sim_measurements,configs[platform],show=False,plot_dir='',
                              plot_flag=False)
    pprint.pprint(metrics)


    #run some measurements and metrics based on filters
    #the community and population scale measurements that are included in scenario1 will be generated
    metrics = run_all_metrics(gt_measurements,sim_measurements,configs[platform],show=False,plot_dir='',
                              plot_flag=False,filters={"scenario1":True,"scale":["community","population"]})
    pprint.pprint(metrics)
                              



def main():

    #baseline_examples("reddit")

    network_examples("twitter")

    #cascade_examples("twitter")


if __name__ == "__main__":
    main()
