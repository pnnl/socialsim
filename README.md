# socialsim

This repo contains scripts needed to run the measurements and metrics for the SocialSim challenge evaluation.

## Change Log

* **12 October 2018**: 
   * We updated the network_measurements implementations to use igraph and SNAP rather than networkx for improved memory and time performance.  Some of our team members had trouble with the python-igraph and SNAP installations.  If you have trouble with the python-igraph installation using pip, try the conda install - "conda install -c conda-forge python-igraph".   SNAP should be installed from https://snap.stanford.edu/snappy/ rather than using pip.  If you get a "Fatal Python error: PyThreadState_Get: no current thread" error, you should modify the SNAP setup.py file and replace "dynlib_path = getdynpath()" with e.g. "dynlib_path = "/anaconda/lib/libpython2.7.dylib" (use the path to your libpython2.7.dylib file).  Please contact us if you are having trouble with your installation after following these steps.  
   * Additionally, we moved from the CSV input format to the JSON input format.  Example JSON files for each platform can be found on the December Challenge wiki page in the same place as the example csv files.

* **9 October 2018**: 
   * We updated the cascade_measurements so that cascade-level measurements are calculated using the CascadeCollectionMeasurements class rathan the SingleCascadeMeasurements class.  This means that all cascade measurements can now be calculated using the CascadeCollectionMeasurements class.  The cascade_examples function shows how to run cascade measurements.  Additionally, we fixed the implementation of the cascade breadth calculation.

## Scripts

### run_measurements_and_metrics.py

This is the main script that provides functionality to run individual measurements and metrics or the full set of assigned measurements and metrics for the challenge (this replaces
the previous metrics_config.py script). 

#### Measurement Configuration

The measurement configurations used by run_measurements_and_metrics.py are found in the metric_config files in the config/ directory.  These 
files define a set of dictionaries for different measurement types that specify the measurement and metric parameters. There are five metrics_config files:

1. network_metrics_config.py - contains `network_measurement_params` to be used for all network measurements
2. cascade_metrics_config.py - contains `cascade_measurement_params` to be used for all cascade measurements
3. baseline_metrics_config_github.py - contains `github_measurement_params` to be used for baseline measurements applied to GitHub
3. baseline_metrics_config_reddit.py - contains `reddit_measurement_params` to be used for baseline measurements applied to Reddit
3. baseline_metrics_config_twitter.py - contains `twitter_measurement_params` to be used for baseline measurements applied to Twitter


Each dictionary element in one of the measurement_params dictionaries defines the metric assignments for a single measurement, with the key indicating the name of the 
measurement and the value specifying the measurement function, measurement function arguments, scenarios which the measurement is included for,
and metrics functions for the metric calculation.
For example, here is the specification of a single measurement in this format:

```python
 measurement_params = {
 "user_unique_repos": {
        'question': '17',
        "scale": "population",
        "node_type":"user",
	"scenario1":True,
	"scenario2":False,
	"scenario3":True,
        "measurement": "getUserUniqueRepos",
	"measurement_args":{"eventType":contribution_events},
        "metrics": { 
            "js_divergence": named_partial(Metrics.js_divergence, discrete=False),
            "rmse": Metrics.rmse,
            "r2": Metrics.r2}
    }
 }   
```

This measurement is related to the number of unique repos that users contribute to (Question #17), which is a user-centric 
measurement at the population level.  The measurement will be used in scenario 1 and scenario 3, but not scenario 2.
The "measurement" keyword specifies the measurement function to  apply, and the "measurement_args" keywords specifies 
the arguments to the measurement function in dictionary format.  The "metrics" keyword provides a dictionary of each of 
the metrics that should be applied for this measurement.

#### Measurements Classes

Measurements are calculated on a data set by employing one of the measurements classes.  There are currently 6 measurements classes which produce different categories of measurements.  
1. BaselineMeasurements implemented in BaselineMeasurements.py - this includes all measurements from the baseline challenge which have been generalized to apply to GitHub,Twitter, or Reddit
2. GithubNetworkMeasurements implemented in network_measurements.py - this includes network measurements for Github.
3. RedditNetworkMeasurements implemented in network_measurements.py - this includes network measurements for Reddit.
4. TwitterNetworkMeasurements implemented in network_measurements.py - this includes network measurements for Twitter.
5. SingleCascadeMeasurements implemented in cascade_measurements.py - this includes node level cascade measurements (i.e. measurements on a single cascade)
6. CascadeCollectionMeasurements implemented in cascade_measurements.py - this includes population and community level cascade measurements (i.e. measurements on a set of cascades)

To instantiate a measurements object for particular data set (either simulation or ground truth data), you generally pass the data frame to one of the above classes:

```python
#create measurement object from data frame 
measurement = BaselineMeasurements(data_frame)
#create measurement object from csv file
measurement = BaselineMeasurements(csv_file_name)

#create measurement object with specific list of nodes to calculate node-level measurements on
measurement = BaselineMeasurements(data_frame,user_node_ids=['user_id1'],content_node_ids=['repo_id1'])
```

This object contains the methods for calculating all of the measurements of the given type.  For example, the user unique repos measurement can be calculated as follows:

```python
result = measurement.getUserUniqueRepos(eventType=contribution_events)
```

#### Running a Single Measurement

The `run_measurement` funciton can be used to calculate the measurement output for a single measurement on a given data set using the measurement_params configuration, which contains the parameters to be used for evaluation during the challenge event.  The arguments for this function include the data, the measurement_params dictionary, and the name of the measurement to apply.

For example, if we want to run one of the baseline GitHub measurements on the simulation data, we need to provide the `github_measurement_params` dictionary which contains the relavent configution and provide the name of the specific measurement we are interested in:

```python
simulation = BaselineMeasurements(simulation_data_frame)
meas = run_measurement(simulation, github_measurement_params, "user_unique_content")
```

The `run_metrics` function can be used to run all the relevant metrics for a given measurement in addition to the measurement output itself.  
This function takes two Measurement objects as input, one for the ground truth and one for the simulation, the relevant measurement_params dictionary, and the name of the measurement as listed in the keywords of measurement_params. It returns the measurement results for the ground truth and the simulation and the metric output.

For example:

```python                                                                                                            
ground_truth = BaselineMeasurements(ground_truth_data_frame)
simulation = BaselineMeasurements(simulation_data_frame)
gt_measurement, sim_measurement, metric = run_metrics(ground_truth, simulation, github_measurement_params, "user_unique_content")
```

#### Running All Measurements

To run the all the  measurements that are defined in the measurement_params configuration, the `run_all_measurements` and `run_all_metrics`
functions can be used.  To run all  the measurements on a simulation data Measurements object and save the output in pickle files in the output directory:

```python
meas_dictionary = run_all_measurements(simulation,github_measurement_params,output_dir='measurement_output/')
```

To run all the metrics for all the measurements on a ground truth Measurements object and simulation data Measurements object:

```python
metrics = run_all_metrics(ground_truth,simulation,github_measurement_params)
```

For both `run_all_metrics` and `run_all_measurements`, you can additionally specify specific subsets of the measurements by using the filter parameter to filter on any properties in the measurement_params dictionary.  For example:

```python
metrics = run_all_metrics(ground_truth,simulation,github_measurement_params,filters={"scale":"population","node_type":"user")
```

#### Plotting

In order to generate plots of the measurements, any of the `run_metrics`, `run_measurement`, `run_all_metrics`, and `run_all_measurements` scripts can take the following arguments:

1. plot_flag - boolean indicator of whether to generate plots
2. show - boolean indicator of whether to display the plots to screen
3. plot_dir - A directory in which to save the plots.  If plot_dir is an empty string '', the plots will not be saved.

Currently, plotting is only implemented for the baseline challenge measurements.  Plotting functionality for the remaining meausrements will be released at a later date.

### Metrics.py

This script contains implementations of each metric for comparison of the output of the ground truth and simulation
measurements.

### BaselineMeasurements.py

This script contains the core BaselineMeasurements class which performs intialization of all input data for measurement calculation
for the measurements from the baseline challenge.

### UserCentricMeasurements.py

This script contains implementations of the user-centric measurements inside the UserCentricMeasurements class.

### ContentCentricMeasurements.py

This script contains implementations of the baseline content-centric measurements inside the ContentCentricMeasurements class.

### CommunityCentricMeasurements.py

This script contains implementations of the community-centric measurements inside the CommunityCentricMeasurements class.

### network_measurements.py

This script contains implementations of the network measurements inside the GithubNetworkMeasurements,RedditNetworkMeasurements, and TwitterNetworkMeasurements classes.

### cascade_measurements.py

This script contains implementations of the cascade measurements inside the SingleCascadeMeasurements and CascadeCollectionMeasurements classes.

