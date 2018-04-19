# socialsim

This repo contains scripts needed to run the measurements and metrics for the SocialSim Baseline Challenge evaluation.

## Scripts

### metrics_config.py

Contains measurement and metric configuration parameters including measurement to metric assignments and provides functionality
to run the metrics for a selected measurement and to run the full set of assigned measurements and metrics.

#### Measurement Configuration

The metric to measurement assignments are defined in the measurement_params dictionary. 
Each dictionary element defines the metric assignments for a single measurement, with the key indicating the name of the 
measurement and the value specifying the filters, measurement function, and metrics functions for the metric calculation.
For example, here is the specification of a single measurement in this format:

```python
 measurement_params = {
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
    }
 }   
```

This measurement is related to the number of unique repos that users contribute to (Question #17), which is a user-centric 
measurement at the population level.  The "filters" keyword specifies that the data should be pre-filtered on the 
"event" field  to include a specified list of event types.  The "measurement" keyword specifies the measurement function to 
apply, while the "metrics" keyword provides a dictionary of each of the metrics that should be applied for this measurement.

#### Running a Single Measurement

The `run_metrics` function can be used to run all the relevant metrics for a given measurement based on the 
measurement_params configuration.  This function takes the ground_truth and simulation data in the 4-column data frame format
("time","eventType","userID","repoID") and the name of the measurement as listed in the keywords of measurement_params. 
For example:

```python                                                                                                            
gt_measurement, sim_measurement, metric = run_metrics(ground_truth, simulation, "repo_contributors")
```

The default behavior for the implementation of the node-level measurements is to select the single most-active node 
for evaluation.  In order to specify which users or repos the node-level measurements should be calculated for, the users and repos 
parameters can be used by providing a list of users (login_h) and repos (full_name_h).  For example,

```python
gt_measurement, sim_measurement, metric = run_metrics(ground_truth, simulation, 
                                                      "user_activity_timeline",
                                                      users=['PeZv4Yha0B_17dV8SAioFA'])
```

#### Running All Measurements

To run the metrics for all the measurements that are defined in the measurement_params configuration, the run_all_metrics
function can be used.  To run all the metrics for all the measurements on a ground truth data frame and simulation data frame:

```python
metrics = run_all_metrics(ground_truth,simulation)
```

You can additionally specify specific subsets of the measurements by scale ("node" or "population") and by node type 
("user" or "repo"):

```python
metrics = run_all_metrics(ground_truth,simulation,scale="population",node_type="user")
```

Additionally, the user and repo arguments for specifying the nodes to use for node-level measurements can be passed to this
function.

```python
metrics = run_all_metrics(ground_truth,simulation,node_type="repo",
                          repos=['3mGSybhub0IE-iZ0nOcOmg/fxFnpSLfvseMwBr1Z3NPkw',
                                 'B8ZJ9zQBfx4zJyuG6QCWcQ/73uOGPnes5YM9RW6Bst3GQ'])
```

### Metrics.py

This script contains implementations of each metric for comparison of the output of the ground truth and simulation
measurements.

### UserCentricMeasurements.py

This script contains implementations of the user-centric measurements which take as input a dataframe in the 4-column format 
("time","eventType","userID","repoID").

### RepoCentricMeasurements.py

This script contains implementations of the repo-centric measurements which take as input a dataframe in the 4-column format 
("time","eventType","userID","repoID").

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

