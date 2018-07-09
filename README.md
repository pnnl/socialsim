# socialsim

This repo contains scripts needed to run the measurements and metrics for the SocialSim Baseline Challenge evaluation.

## Scripts

Update 07/06/18: The up-to-date versions of the measurements and metrics scripts can be found in github-measurements.  The previously released version is provided in github-measurement-old.  The below instructions correspond to the version in github-measurements. This new version requires several PKL files which are hosted on the Metrics Release v1 page on the SocialSim wiki:

1. communities.pkl
2. filtRepos-test.pkl﻿
3. filtUsers-test.pkl﻿

These files should be downloaded and placed in the directory github-measurements/data/. 

Another major change is that the `run_metrics` and `run_all_metrics` functions no longer take data frames as input but instead take Measurement objects.  Examples of how to call these functions can be found below or in the main function of metrics_config.py.

### metrics_config.py

Contains measurement and metric configuration parameters including measurement to metric assignments and provides functionality
to run the metrics for a selected measurement and to run the full set of assigned measurements and metrics.

#### Measurement Configuration

The metric to measurement assignments are defined in the measurement_params dictionary. 
Each dictionary element defines the metric assignments for a single measurement, with the key indicating the name of the 
measurement and the value specifying the measurement function, measurement function arguments, and metrics functions for the metric calculation.
For example, here is the specification of a single measurement in this format:

```python
 measurement_params = {
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
    }
 }   
```

This measurement is related to the number of unique repos that users contribute to (Question #17), which is a user-centric 
measurement at the population level.  The "measurement" keyword specifies the measurement function to 
apply, and the "measurement_args" keywords specifies the arguments to the measurement function in dictionary format.  The "metrics" keyword provides a dictionary of each of the metrics that should be applied for this measurement.

#### Measurements Class

Measurements are calculated on a data set by employing the Measurements class (implemented in Measurements.py).  To instantiate a measurements object for particular data set (either simulation or ground truth data):

```python
#create measurement object from data frame 
measurement = Measurement(data_frame)
#create measurement object from csv file
measurement = Measurement(csv_file_name)

#create measurement object with specific list of nodes to calculate node-level measurements on
measurement = Measurement(data_frame,interested_users=['user_id1'],interested_repos=['repo_id1'])
```

This object contains the methods for calculating all of the measurements.  For example, the user unique repos measurement can be calculated as follows:

```python
result = measurement.getUserUniqueRepos(eventType=contribution_events)
```

#### Running a Single Measurement

The `run_metrics` function can be used to run all the relevant metrics for a given measurement based on the 
measurement_params configuration, which contains the parameters to be used for evaluation during the challenge event.  This function takes two Measurement objects as input, one for the ground truth and one for the simulation, and the name of the measurement as listed in the keywords of measurement_params. It returns the measurement results for the ground truth and the simulation and the metric comparison.

For example:

```python                                                                                                            
ground_truth = Measurement(ground_truth_data_frame)
simulation = Measurement(simulation_data_frame)
gt_measurement, sim_measurement, metric = run_metrics(ground_truth, simulation, "repo_contributors")
```

If the Measurement objects do not have the interested_users and interested_repos keywords set, then there will be no node-level measurements calculated.

#### Running All Measurements

To run the metrics for all the measurements that are defined in the measurement_params configuration, the run_all_metrics
function can be used.  To run all the metrics for all the measurements on a ground truth Measurements object and simulation data Measurements object:

```python
metrics = run_all_metrics(ground_truth,simulation)
```

You can additionally specify specific subsets of the measurements by scale ("node" or "population") and by node type 
("user" or "repo"):

```python
metrics = run_all_metrics(ground_truth,simulation,scale="population",node_type="user")
```

### Metrics.py

This script contains implementations of each metric for comparison of the output of the ground truth and simulation
measurements.

### Measurements.py

This script contains the core Measurements class which performs intialization of all input data for measurement calculation.

### UserCentricMeasurements.py

This script contains implementations of the user-centric measurements inside the UserCentricMeasurements class.

### RepoCentricMeasurements.py

This script contains implementations of the repo-centric measurements inside the RepoCentricMeasurements class.

### CommunityCentricMeasurements.py

This script contains implementations of the community-centric measurements inside the CommunityCentricMeasurements class.

## Old Scripts

### TransferEntropy.py

This script contains functionality for calculating the transfer entropy between pairs of time series for the 
influence measurements.  The full calculation of the influence measurements and metrics will be added in a future
update.

### RepoMeasurementsWithPlot.py (DEPRECATED)

This script contains implementations of the repo-centric measurements along with corresponding plotting functionality to 
visualize the resulting measurements.  However, the measurement implementations here are not up-to-date.  The up-to-date
implementations are in RepoCentricMeasurements.py.  The plotting functionality will be merged into RepoCentricMeasurements.py
in a future update.

### UserMeasurementsWithPlot.py (DEPRECATED)

This script contains implementations of the user-centric measurements along with corresponding plotting functionality to 
visualize the resulting measurements.  However, the measurement implementations here are not up-to-date.  The up-to-date
implementations are in UserCentricMeasurements.py.  The plotting functionality will be merged into UserCentricMeasurements.py
in a future update.

### plots.py

Helper functions for plots visualizing the output of the measurements code.

