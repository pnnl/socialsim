import Metrics
from run_measurements_and_metrics import named_partial

network_measurement_params = {
    ### Github
    "number_of_nodes": {
        "question": '',
        "scale": "population",
        "scenario1":True,
        "scenario2":False,
        "sceanrio2":True,
        "measurement": "number_of_nodes",
        "metrics": {
            "absolute_percentage_error": Metrics.absolute_percentage_error,
        }
    },

    "number_of_edges": {
        "question": '',
        "scale": "population",
        "scenario1":True,
        "scenario2":False,
        "sceanrio2":True,
        "measurement": 'number_of_edges',
        "metrics": {
            "absolute_percentage_error": Metrics.absolute_percentage_error,
        }
    },

    "density": {
        "question": '',
        "scale": "population",
        "scenario1":True,
        "scenario2":False,
        "sceanrio2":True,
        "measurement": 'density',
        "metrics": {
            "absolute_percentage_error": Metrics.absolute_percentage_error,
        }
    },

    "shortest_path_length_distribution": {
        "question": '',
        "scale": "population",
        "scenario1":True,
        "scenario2":False,
        "sceanrio2":True,
        "measurement": 'shortest_path_length_distribution',
        "metrics": {
            "js_divergence": named_partial(Metrics.js_divergence, discrete=True),
        }
    },

    "min_shortest_path_length": {
        "question": '',
        "scale": "population",
        "scenario1":True,
        "scenario2":False,
        "sceanrio2":True,
        "measurement": 'min_shortest_path_length',
        "metrics": {
            "absolute_percentage_error": Metrics.absolute_percentage_error,
        }
    },
    "max_shortest_path_length": {
        "question": '',
        "scale": "population",
        "scenario1":True,
        "scenario2":False,
        "sceanrio2":True,
        "measurement": 'max_shortest_path_length',
        "metrics": {
            "absolute_percentage_error": Metrics.absolute_percentage_error,
        }
    },
    "mean_shortest_path_length": {
        "question": '',
        "scale": "population",
        "scenario1":True,
        "scenario2":False,
        "sceanrio2":True,
        "measurement": 'mean_shortest_path_length',
        "metrics": {
            "absolute_percentage_error": Metrics.absolute_percentage_error,
        }
    },

    "assortativity_coefficient": {
        "question": '',
        "scale": "population",
        "scenario1":True,
        "scenario2":False,
        "sceanrio2":True,
        "measurement": 'assortativity_coefficient',
        "metrics": {
            "absolute_percentage_error": Metrics.absolute_percentage_error,
            "absolute_difference": Metrics.absolute_difference,
        }
    },

    "number_of_connected_components": {
        "question": '',
        "scale": "population",
        "scenario1":True,
        "scenario2":False,
        "sceanrio2":True,
        "measurement": 'number_of_connected_components',
        "metrics": {
            "absolute_percentage_error": Metrics.absolute_percentage_error,
        }
    },

    "diameter_of_largest_connected_components": {
        "question": '',
        "scale": "population",
        "scenario1":True,
        "scenario2":False,
        "sceanrio2":True,
        "measurement": 'diameter_of_largest_connected_components',
        "metrics": {
            "absolute_percentage_error": Metrics.absolute_percentage_error,
        }
    },

    "average_clustering_coefficient": {
        "question": '',
        "scale": "population",
        "scenario1":True,
        "scenario2":False,
        "sceanrio2":True,
        "measurement": 'average_clustering_coefficient',
        "metrics": {
            "absolute_percentage_error": Metrics.absolute_percentage_error,
            "absolute_difference": Metrics.absolute_difference,
        }
    },

    "min_node_degree": {
        "question": '',
        "scale": "population",
        "scenario1":True,
        "scenario2":False,
        "sceanrio2":True,
        "measurement": 'min_node_degree',
        "metrics": {
            "absolute_percentage_error": Metrics.absolute_percentage_error,
        }
    },
    "max_node_degree": {
        "question": '',
        "scale": "population",
        "scenario1":True,
        "scenario2":False,
        "sceanrio2":True,
        "measurement": 'max_node_degree',
        "metrics": {
            "absolute_percentage_error": Metrics.absolute_percentage_error,
        }
    },
    "mean_node_degree": {
        "question": '',
        "scale": "population",
        "scenario1":True,
        "scenario2":False,
        "sceanrio2":True,
        "measurement": 'mean_node_degree',
        "metrics": {
            "absolute_percentage_error": Metrics.absolute_percentage_error,
        }
    },

    "degree_distribution": {
        "question": '',
        "scale": "population",
        "scenario1":True,
        "scenario2":False,
        "sceanrio2":True,
        "measurement": 'degree_distribution',
        "metrics": {
            "js_divergence": named_partial(Metrics.js_divergence, discrete=True),
        }
    },

    "page_rank_distribution": {
        "question": '',
        "scale": "population",
        "scenario1":True,
        "scenario2":False,
        "sceanrio2":True,
        "measurement": 'page_rank_distribution',
        "metrics": {
            "js_divergence": named_partial(Metrics.js_divergence, discrete=True),
        }
    },

    "community_structure": {
        "question": '',
        "scale": "population",
        "scenario1":True,
        "scenario2":False,
        "sceanrio2":True,
        "measurement": 'community_structure',
        "metrics": {
            "absolute_percentage_error": Metrics.absolute_percentage_error,
            "absolute_difference": Metrics.absolute_difference,
        }
    },
}
