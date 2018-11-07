measurement_plot_params = {

    ### community

    "community_burstiness": {
        "data_type": "dict",
        "x_axis": "Community",
        "y_axis": "Burstiness",
        "plot": ['bar']
    },

    "community_contributing_users": {
        "data_type": "dict",
        "x_axis": "Community",
        "y_axis": "Proportion of Users Contributing",
        "plot": ['bar']
    },

    "community_event_proportions": {
        "data_type": "dict_DataFrame",
        "x_axis": "Event Type",
        "y_axis": "Event Proportion",
        "plot": ['bar'],
        "plot_keys": "community"
    },

    "community_geo_locations": {
        "data_type": "dict_DataFrame",
        "x_axis": "Country",
        "y_axis": "Number of Events",
        "plot": ['bar'],
        "plot_keys": "community"
    },

    "community_issue_types": {  # result None type
        "data_type": "dict_DataFrame",
        "x_axis": "Date",
        "y_axis": "Number of Issues",
        "plot": ['multi_time_series'],
        "plot_keys": "community"

    },

    "community_num_user_actions": {
        "data_type": "dict_DataFrame",
        "x_axis": "Date",
        "y_axis": "Mean Number of User Actions",
        "hue": "Key",
        "plot": ['time_series'],
        "plot_keys": "community_subsets"
    },
    #

    'community_user_account_ages': {
        "data_type": "dict_Series",
        "x_axis": "User Account Age",
        "y_axis": "Number of Actions",
        "plot": ['hist'],
        "plot_keys": "community"
    },

    'community_user_burstiness': {
        "data_type": "dict_Series",
        "x_axis": "User Burstiness",
        "y_axis": "Number of Users",
        "plot": ['hist'],
        "plot_keys": "community"
    },

    #
    "community_gini": {
        "data_type": "dict",
        "x_axis": "Community",
        "y_axis": "Gini Scores",
        "plot": ['bar']
    },

    "community_palma": {
        "data_type": "dict",
        "x_axis": "Community",
        "y_axis": "Palma Scores",
        "plot": ['bar']
    },

    # repo
    #

    "content_contributors": {
        "data_type": "dict_DataFrame",
        "x_axis": "Date",
        "y_axis": "Number of Contributors",
        "plot": ['time_series'],
        "plot_keys": "content"
    },

    "content_diffusion_delay": {
        "data_type": "dict_Series",
        "x_axis": "Diffusion Delay",
        "y_axis": "Number of Events",
        "plot": ['hist'],
        "plot_keys": "content"
    },

    "repo_event_counts_issue": {
        "data_type": "DataFrame",
        "y_axis": "Number of Repos",
        "x_axis": "Number of Issue Events",
        "plot": ['hist']
    },

    "repo_event_counts_pull_request": {
        "data_type": "DataFrame",
        "y_axis": "Number of Repos",
        "x_axis": "Number of Pull Requests",
        "plot": ['hist']
    },

    "repo_event_counts_push": {
        "data_type": "DataFrame",
        "y_axis": "Number of Repos",
        "x_axis": "Number of Push Events",
        "plot": ['hist']
    },

    "content_event_distribution_daily": {
        "data_type": "dict_DataFrame",
        "x_axis": "Date",
        "y_axis": "# Events",
        "plot": ['multi_time_series'],
        "plot_keys": "content"
    },

    "content_event_distribution_dayofweek": {
        "data_type": "dict_DataFrame",
        "x_axis": "Day of Week",
        "y_axis": "# Events",
        "plot": ['multi_time_series'],
        "plot_keys": "content"
    },

    "content_growth": {
        "data_type": "dict_DataFrame",
        "x_axis": "Date",
        "y_axis": "# Events",
        "plot": ['time_series'],
        "plot_keys": "content"
    },
    #
    "repo_issue_to_push": {
        "data_type": "dict_DataFrame",
        "x_axis": "Number of Previous Events",
        "y_axis": "Issue Push Ratio",
        "plot": ['time_series'],
        "plot_keys": "content"
    },

    "content_liveliness_distribution": {
        "data_type": "DataFrame",
        "y_axis": "Number of Repos/Posts/Tweets",
        "x_axis": "Number of Forks/Comments/Replies",
        "plot": ['hist']
    },

    "repo_trustingness": {
        "data_type": "DataFrame",
        "x_axis": "Ground Truth",
        "y_axis": "Simulation",
        "plot": ['scatter']
    },

    "content_popularity_distribution": {
        "data_type": "DataFrame",
        "y_axis": "Number of Repos/Tweets",
        "x_axis": "Number of Watches/Rewtweets",
        "plot": ['hist']
    },

    "repo_user_continue_prop": {
        "data_type": "dict_DataFrame",
        "x_axis": "Number of Actions",
        "y_axis": "Probability of Continuing",
        "plot": ['time_series'],
        "plot_keys": "content"
    },
    #
    #
    # ### user

    "user_popularity": {
        "data_type": "DataFrame",
        "y_axis": "Number of Users",
        "x_axis": "Popularity of User's Repos/Tweets/Posts",
        "plot": ['hist']
    },

    "user_activity_distribution": {
        "data_type": "DataFrame",
        "x_axis": "User Activity",
        "y_axis": "Number of Users",
        "plot": ['hist']
    },

    "user_diffusion_delay": {
        "data_type": "Series",
        "x_axis": "Diffusion Delay (H)",
        "y_axis": "Number of Events",
        "plot": ['hist']
    },
    "user_activity_timeline": {
        "data_type": "dict_DataFrame",
        "x_axis": "Date",
        "y_axis": "Number of Events",
        "plot": ['time_series'],
        "plot_keys": "user"
    },

    "user_trustingness": {
        "data_type": "DataFrame",
        "x_axis": "Ground Truth",
        "y_axis": "Simulation",
        "plot": ['scatter']
    },

    "user_unique_content": {
        "data_type": "DataFrame",
        "x_axis": "Number of Unique Repos/Posts/Tweets",
        "y_axis": "Number of Users",
        "plot": ['hist']
    }
}

cascade_measurement_plot_params = {
    'cascade_breadth_by_depth': {
        'data_type': 'dict_DataFrame',
        'plot': ['time_series'],
        'x_axis': 'Depth',
        'y_axis': 'Breadth',
        'plot_keys':'cascade'},

    'cascade_breadth_by_time':
        {'data_type': 'dict_DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Date',
         'y_axis': 'Breadth',
         'plot_keys':'cascade'},

    'cascade_max_depth_over_time':
        {'data_type': 'dict_DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Date',
         'y_axis': 'Depth',
         'plot_keys':'cascade'},

    'cascade_new_user_ratio_by_depth':
        {'data_type': 'dict_DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Depth',
         'y_axis': 'New User Ratio',
         'plot_keys':'cascade'},

    'cascade_new_user_ratio_by_time':
        {'data_type': 'dict_DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Date',
         'y_axis': 'New User Ratio',
         'plot_keys':'cascade'},

    'cascade_size_over_time':
        {'data_type': 'dict_DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Date',
         'y_axis': 'Cascade Size',
         'plot_keys':'cascade'},

    'cascade_structural_virality_over_time':
        {'data_type': 'dict_DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Date',
         'y_axis': 'Structural Virality',
         'plot_keys':'cascade'},

    'cascade_uniq_users_by_depth':
        {'data_type': 'dict_DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Depth',
         'y_axis': 'Unique Users',
         'plot_keys':'cascade'},

    'cascade_uniq_users_by_time':
        {'data_type': 'dict_DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Date',
         'y_axis': 'Unique Users',
         'plot_keys':'cascade'},

    'community_cascade_lifetime_distribution':
        {'data_type': 'dict_DataFrame',
         'plot': ['hist'],
         'x_axis': 'Lifetime',
         'y_axis': 'Number of Cascades',
         'plot_keys':'community'},

    'community_cascade_lifetime_timeseries':
        {'data_type': 'dict_DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Date',
         'y_axis': 'Cascade Lifetime',
         'plot_keys':'community'},

    'community_cascade_size_distribution':
        {'data_type': 'dict_DataFrame',
         'plot': ['hist'],
         'x_axis': 'Size',
         'y_axis': 'Number of Cascades',
         'plot_keys':'community'},

    'community_cascade_size_timeseries':
        {'data_type': 'dict_DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Time',
         'y_axis': 'Cascade Size',
         'plot_keys':'community'},

    'community_max_breadth_distribution':
        {'data_type': 'dict_DataFrame',
         'plot': ['hist'],
         'x_axis': 'Max Breadth',
         'y_axis': 'Number of Cascades',
         'plot_keys':'community'},

    'community_max_depth_distribution':
        {'data_type': 'dict_DataFrame',
         'plot': ['hist'],
         'x_axis': 'Max Depth',
         'y_axis': 'Number of Cascades',
         'plot_keys':'community'},

    'community_new_user_ratio_by_time':
        {'data_type': 'dict_DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Date',
         'y_axis': 'New User Ratio',
         'plot_keys':'community'},

    'community_structural_virality_distribution':
        {'data_type': 'dict_DataFrame',
         'plot': ['hist'],
         'x_axis': 'Structural Virality',
         'y_axis': 'Number of Cascade',
         'plot_keys':'community'},

    'community_unique_users_by_time':
        {'data_type': 'dict_DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Date',
         'y_axis': 'Unique Users',
         'plot_keys':'community'},

    'population_cascade_lifetime_distribution':
        {'data_type': 'DataFrame',
         'plot': ['hist'],
         'x_axis': 'Cascade Lifetime',
         'y_axis': 'Number of Cascades'},

    'population_cascade_lifetime_timeseries':
        {'data_type': 'DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Date',
         'y_axis': 'Cascade Lifetime'},

    'population_cascade_size_distribution':
        {'data_type': 'DataFrame',
         'plot': ['hist'],
         'x_axis': 'Size',
         'y_axis': 'Number of Cascades'},

    'population_cascade_size_timeseries':
        {'data_type': 'DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Date',
         'y_axis': 'Cascade Size'},

    'population_max_breadth_distribution':
        {'data_type': 'DataFrame',
         'plot': ['hist'],
         'x_axis': 'Max Breadth',
         'y_axis': 'Number of Cascades'},

    'population_max_depth_distribution':
        {'data_type': 'DataFrame',
         'plot': ['hist'],
         'x_axis': 'Max Depth',
         'y_axis': 'Number of Cascades'},

    'population_structural_virality_distribution':
        {'data_type': 'DataFrame',
         'plot': ['hist'],
         'x_axis': 'Structural Virality',
         'y_axis': 'Number of Cascade'}
}

measurement_plot_params.update(cascade_measurement_plot_params)
