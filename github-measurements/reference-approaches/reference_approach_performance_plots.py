
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import json
from metrics_config import *
import numpy as np

import seaborn as sns
import glob

import collections
import pprint

fields = {'metric':'Metric',
'rq':'RQ',
'node_type':'Node Type',
'scale':'Scale',
'temporal':'Temporal'}

pal = sns.color_palette()

direction_map = {'abs':-1,
                'rbo':1,
                'r2':1,
                'ks':-1,
                'js':-1,
                'dtw':-1,
                'rmse':-1}


def load_results(results_dir):

    """
    Read in all metrics JSON files from a directory
    """

    count = 0
    results = []
    for fn in glob.glob(results_dir + '/*.json'):
        with open(fn,'r') as f:
            res = json.load(f)
        results.append(res)


    metrics = []
    for res in results:
        metrics.append(flatten(res))

    df = pd.DataFrame(metrics)
    df_tall = reformat_df(df)
    df_tall = measurement_types(df_tall)
    
    return df,df_tall


def flatten(d, parent_key='', sep='-'):

    """
    Flatten nested dictionary
    https://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys
    """

    items = []
    for k, v in d.items():
      new_key = parent_key + sep + k if parent_key else k
      if isinstance(v, collections.MutableMapping):
        items.extend(flatten(v, new_key, sep=sep).items())
      else:
        items.append((new_key, v))
    return dict(items)



def reformat_df(df_wide):

    """
    Covert metrics data frame from wide format (one column per metric) to tall format (only one column with metrics)
    """

    meta_cols = ['event_count-event_count','institution','latest_event-latest_event','version','uuid','source','team',
                 'dataset','submission_time']
    meas_cols = [c for c in df_wide.columns if c not in meta_cols]

    df_tall = df_wide.melt(id_vars=meta_cols)

    df_tall['measurement'] = df_tall['variable'].apply(lambda x: x.split('-')[0])
    df_tall['metric'] = df_tall['variable'].apply(lambda x: x.split('-')[1])

    #clean up non-numerics
    df_tall = df_tall.fillna(value=np.nan)
    df_tall = df_tall[~df_tall['metric'].isin(['error','exception'])]

    #average over models
    mean_df_tall = df_tall.groupby(['version','team','institution','variable'])['value'].apply(lambda x: np.mean(x)).reset_index()


    mean_df = mean_df_tall.pivot_table(values='value',
                                       index=['team','version','institution'],columns='variable').reset_index()
    mean_df_tall['measurement'] = mean_df_tall['variable'].apply(lambda x: x.split('-')[0])
    mean_df_tall['metric'] = mean_df_tall['variable'].apply(lambda x: x.split('-')[1])

    return mean_df_tall
    

def read_reference(fn,ra_name):
    
    """
    Read JSON file with reference approach metrics results
    """

    with open(fn,'r') as f:
        ra = json.load(f)

    values = {}
    comm_meas = []
    for key in ra:
        if (not ra[key].keys()[0] in ['r2','js_divergence','ks_test','rmse','dtw','rbo','absolute difference','absolute_difference']) and (len(ra[key].keys()[0]) <= 20):
            values[key] = {}
            for c in ra[key].keys():
                for m in ra[key][c].keys():
                    if m in values[key].keys():
                        values[key][m].append(ra[key][c][m])
                    else:
                        values[key][m] = [ra[key][c][m]]

            comm_meas.append(key)

    for key in comm_meas:
        del ra[key]
        
    for key in values:
        for m in values[key]:
            values[key][m] = np.mean([v for v in values[key][m] if not v is None])
    community_baseline = values

    values = {}
    node_meas = []
    for key in ra:
        if len(ra[key].keys()[0]) >= 20:
            values[key] = {}
            for c in ra[key].keys():
                for m in ra[key][c].keys():
                    if m in values[key].keys():
                        values[key][m].append(ra[key][c][m])
                    else:
                        values[key][m] = [ra[key][c][m]]
            node_meas.append(key)

    for key in node_meas:
        del ra[key]

    for key in values:
        for m in values[key]:
            values[key][m] = np.mean([v for v in values[key][m] if not v is None])
    node_baseline = values

    ra.update(node_baseline)
    ra.update(community_baseline)

    with open(fn.replace('.json','_processed.json'),'w') as f:
        json.dump(ra,f)
    
    ra = pd.DataFrame(ra).T

    ra = ra.reset_index().melt(id_vars='index').dropna()
    ra.columns = ['measurement','metric','value']
    ra['ra_name'] = ra_name
    
    ra['metric'] = ra['metric'].replace({'ks_test':'ks',
                                        'absolute difference':'abs',
                                        'absolute_difference':'abs',
                                        'js_divergence':'js'})
    ra['measurement'] = ra['measurement'].replace({'most_active_users':'user_get_most_active',
                                             'user_total':'te_user_total',
                                               'repo_total':'te_repo_total',
                                              'user_interactions':'te_user_interactions',
                                              'repo_interactions':'te_repo_interactions'})
    return(ra)



def get_measurement_info(x,field):
    if str(x) in measurement_params.keys():
        return measurement_params[str(x)][field]
    else:
        return ''


def measurement_types(mean_df_tall):

    mean_df_tall['scale'] = mean_df_tall['measurement'].apply(lambda x: get_measurement_info(x,'scale'))
    mean_df_tall['rq'] = mean_df_tall['measurement'].apply(lambda x: get_measurement_info(x,'rq'))
    mean_df_tall['node_type'] = mean_df_tall['measurement'].apply(lambda x: get_measurement_info(x,'node_type'))
    mean_df_tall['temporal'] = mean_df_tall['measurement'].apply(lambda x: get_measurement_info(x,'temporal'))


    return mean_df_tall


def plot_bar_plot(df,x,y,hue,dodge,estimator,color_dict,xlabel,
                  title,fn,log=True,small_labels=True,
                 ):

    """
    Make bar chart from data frame

    Input:
    df - Simulation metrics data frame in tall format
    x - Column to plot on the x-axis
    y - Column to plot on the y-axis
    dodge - Dodge parameter of seaborn barplot
    estimator - Estimator parameter of seaborn barplot (aggregation function)
    color_dict - Color dictionary for plotting
    xlabel - X-axis label
    title - Plot title
    fn - File name for saving the plot
    log - Boolean flag for log scale
    small_labels - Boolean flag to reduc the font size of the x-tick labels.  Useful when plotting the full set of metrics.
    """

    sns.set_context("talk",font_scale=1.4)
    fig = plt.figure(figsize=(25,18))
    
    values = df[hue].unique()
    values.sort()
    
    if len(color_dict) > 0:
        ax = sns.barplot(data=df,x=x,y=y,hue=hue,dodge=dodge,estimator=estimator,palette=color_dict)
        ax.legend_.remove()
    else:
        sns.barplot(data=df,x=x,y=y,hue=hue,dodge=dodge,estimator=estimator,hue_order=values)
    
    plt.xticks(rotation=90)
    if log:
        plt.yscale('symlog')
    plt.xlabel(xlabel)
    plt.ylabel('Relative Performanace')
    plt.title(title)
    if small_labels:
        plt.xticks(fontsize=14)
    
    if len(color_dict) > 0:
        handles,labels = ax.get_legend_handles_labels()
        sort_idx = sorted(range(len(labels)), key=lambda k: labels[k])
        labels = [labels[s] for s in sort_idx]
        handles = [handles[s] for s in sort_idx]
        ax.legend(handles,labels,loc='best')
    
    plt.tight_layout()
    plt.savefig(fn)


def make_swarm_plot(df,metric,ra_list=[],color_field='version',lower_lim=None,upper_lim=None,log=False):

    """
    Plot swarm plot showing the metric values for each inidividual simulation model for a specific metric type (e.g. RMSE, R2, etc.)

    Inputs:
    df - Simulations metrics data frame (tall format)
    ra_list - List of reference approach data frames to include on the plot. If an empty list, no reference approaches will be plotted.
    color_field - Field of the data frame to use for the color of the data points.  Default is the model version.
    lower_lim - Lower limit of the metric to include in the plot
    upper_lim - Upper limit of the metric to include in the plot
    log - Boolean flag to plot in log scale
    """

    field = 'value'

    sns.set_context("talk",font_scale=2.0)
    plt.figure(figsize=(20,15))
        
    #select metric type of interest
    subset = df[ (df['metric'] == metric) & (~df[field].isnull()) ][['measurement',field,color_field]]
    subset[field] = subset[field].astype(float)

    #filter data based on upper and lower limit
    if not lower_lim is None:
        subset = subset[subset[field] > lower_lim]
        
    if not upper_lim is None:
        subset = subset[subset[field] < upper_lim]
        
    
    #sort data based on mean value of the metric to determine plotting order
    mean_field = subset[['measurement',field]].groupby('measurement').mean().reset_index()
    mean_field.columns = ['measurement','mean_' + field]
    subset = subset.merge(mean_field,on='measurement')
    subset = subset.sort_values('mean_' + field,ascending=False)

    if log:
        plt.yscale('symlog')
    
    sns.swarmplot(y=field,x='measurement',hue=color_field,s=14,data=subset,
                dodge=False)
    plt.xticks(rotation=45)
    
    locs, labels = plt.xticks()  
        
    #plot reference approaches as horizontal lines
    first = {'-':True,'--':True,':':True}
    for ra in ra_list:
        for l in range(len(locs)):

            label = ra['ra_name'].unique()[0]
            if 'January' in label:
                s = '-'
                if 'Sampled' in label:
                    s = '--'
                if 'Random' in label:
                    s = ':'
                try:
                    value = ra[ (ra['measurement'] == labels[l]._text) & (ra['metric'] == metric) ]['value'].values[0]
                    min_x = (locs[l]+0.5)/float(len(locs)) - 0.3 *1./float(len(locs))
                    max_x = (locs[l]+0.5)/float(len(locs)) + 0.3 *1./float(len(locs))
                    if first[s]:
                        plt.axhline(value,xmin=min_x,xmax=max_x,zorder=10,linewidth=5,color='k',
                                   label=label.replace(' January',''),linestyle=s)
                        first[s] = False
                    else:
                        plt.axhline(value,xmin=min_x,xmax=max_x,zorder=10,linewidth=5,color='k',
                                      linestyle=s)
                except:
                    ''

    plt.title(metric + ': Metric Ranges')
    plt.ylabel(metric)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("swarmplot_" + metric + '.png')


def reference_approach_comparison(df,reference_approach,reference_approach_name):

    """
    Make bar charts of performance relative to a given reference approach.

    Inputs:
    df - Simulation metrics data frame
    reference_approach - Reference approach metrics data frame
    reference_approach_name - Name of the reference approach
    """


    reference_approach.columns = ['measurement', 'metric', 'value_ra','ra_name']
    df = df.merge(reference_approach[['measurement','metric','value_ra']],on=['measurement','metric'])

    #calculate relative performance taking into account different directions of metrics (e.g. higher better for R2, lower better for RMSE)
    df['value_rel_ra'] = (df['value'] - df['value_ra']) / df['value_ra'].abs()
    df['metric_direction'] = df['metric'].apply(lambda x: direction_map[x])
    df['value_rel_ra'] = df['metric_direction']*df['value_rel_ra']

    df['meas_metric'] = df['measurement'] + '-' + df['metric']

    #sort df by median value of metrics
    grouped = df.groupby('meas_metric')['value_rel_ra'].median().reset_index()
    grouped.columns = ['meas_metric','mean_value']
    df = df.merge(grouped,on=['meas_metric'],how='left')
    df = df.sort_values('mean_value')

    #remove bad values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['value_rel_ra'])

    color_dicts = {}
    for key in fields:
        values = df[key].unique()
        values.sort()
        color_dicts[key] = {}
        for i,v in enumerate(values):
            color_dicts[key][v] = pal[i]


    #for each grouping of metrics (e.g. node vs. community vs. population, temporal vs. batch, etc.)
    for f in fields:
        #plot each metric individually, averaging over multiple models
        plot_bar_plot(df,'meas_metric','value_rel_ra',f,False,np.median,color_dicts[f],"Measurement and Metric",
                      'Relative Performance By ' + fields[f] + ': ' + reference_approach_name + ' Reference Approach',
                      '_'.join(reference_approach_name.lower().split(' ')) + '_reference_approach_performance_by_' + f + '.png')

        #plot each model invidually, averaging over groups of metrics
        plot_bar_plot(df,f,'value_rel_ra','version',True,np.mean,{},fields[f],
                        'Relative Performance By Model and ' + fields[f] + ': ' + reference_approach_name + ' Reference Approach',
                        '_'.join(reference_approach_name.lower().split(' ')) + '_reference_approach_performance_by_model_and_' + f + '.png',
                     log=True,
                     small_labels=False)


def main():

    #read in all metrics JSON data in the directory
    df_wide,df_tall = load_results('metrics_results/baseline-20180719*.json')

    #read in each reference approach of interest
    #reference approach metrics JSON files can be found on the PNNL wiki page
    ra1 = read_reference('reference_approach_metrics/sampled_january_reference_metrics.json',ra_name='Sampled January')
    ra2 = read_reference('reference_approach_metrics/sampled_august_reference_metrics.json',ra_name='Sampled August')
    ra3 = read_reference('reference_approach_metrics/shifted_august_reference_metrics.json',ra_name='Shifted August')
    ra4 = read_reference('reference_approach_metrics/shifted_january_reference_metrics..json',ra_name='Shifted January')
    ra5 = read_reference('reference_approach_metrics/random_january_reference_metricsx.json',ra_name='Random January')
    ra_list = [ra1,ra2,ra3,ra4,ra5]

    #make plots of relative performance compared with one reference approach
    reference_approach_comparison(df_tall,ra1,'Sampled January')

    #make a swarm plot with reference approaches
    make_swarm_plot(df_tall,"js",ra_list=ra_list)
    
    #make a swarm plot without refrence approaches
    make_swarm_plot(df_tall,"rbo",ra_list=[])


if __name__ == "__main__":
    main()
