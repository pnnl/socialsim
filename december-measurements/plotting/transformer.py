import pandas as pd

def to_DataFrame(data_type):
    data_mapping = {
        'dict': convert_dict,
        'DataFrame': convert_DataFrame,
        'dict_DataFrame': convert_dict_DataFrame,
        'dict_Series': convert_dict_Series,
        'dict_array':convert_dict_array,
        'Series':convert_Series,
        'tuple': None
    }

    return data_mapping.get(data_type, None)




def convert_Series(ground_truth_data=None, sim_data=None, **kwargs):

    if not ground_truth_data is None and not sim_data is None:
        result_df = pd.concat([ground_truth_data.reset_index(drop=True),sim_data.reset_index(drop=True)], axis=1)
        result_df.columns = ['Ground Truth', 'Simulation']
    elif not ground_truth_data is None:
        result_df = pd.DataFrame(ground_truth_data.reset_index(drop=True))
        result_df.columns = ['Ground Truth']
    elif not sim_data is None:
        result_df = pd.DataFrame(sim_data.reset_index(drop=True))
        result_df.columns = ['Simulation']

    return result_df



def convert_dict(ground_truth_data=None, sim_data=None, **kwargs):


    if not ground_truth_data is None and not sim_data is None:
        keys = list(ground_truth_data.keys()) + list(sim_data.keys())

        keys = set(keys)

        data = []
        for k in keys:
            data.append({'Key': k, 'Ground Truth': ground_truth_data.get(k, None), 'Simulation': sim_data.get(k, None)})

        df= pd.DataFrame(data)[['Key','Ground Truth','Simulation']]


    elif not ground_truth_data is None:
        keys = list(ground_truth_data.keys())

        keys = set(keys)

        data = []
        for k in keys:
            data.append({'Key': k, 'Ground Truth': ground_truth_data.get(k, None)})

        df= pd.DataFrame(data)[['Key','Ground Truth']]

    elif not sim_data is None:
        keys = list(sim_data.keys())

        keys = set(keys)

        data = []
        for k in keys:
            data.append({'Key': k, 'Simulation': sim_data.get(k, None)})

        df= pd.DataFrame(data)[['Key','Simulation']]

    return df


def convert_DataFrame(ground_truth_data=None, sim_data=None, **kwargs):
 
    if ground_truth_data is None:
        result_df = sim_data.copy()
        result_df.rename(index=str,columns={'value':'Simulation'},inplace=True)
    elif sim_data is None:
        result_df = ground_truth_data.copy()
        result_df.rename(index=str,columns={'value':'Ground Truth'},inplace=True)
    else:
        merge_cols = [c for c in ground_truth_data.columns if c != 'value']
        result_df = pd.merge(ground_truth_data, sim_data, on=merge_cols, how='outer')
        result_df.columns = merge_cols + ['Ground Truth', 'Simulation']

    return result_df


def convert_dict_DataFrame(ground_truth_data=None, sim_data=None, **kwargs):

    if kwargs.get('key'):

        if not ground_truth_data is None and not sim_data is None and kwargs.get('key') in ground_truth_data and kwargs.get('key') in sim_data:
            merge_columns = [c for c in ground_truth_data[kwargs.get('key')].columns if c != 'value']
            result_df = pd.merge(ground_truth_data[kwargs.get('key')], sim_data[kwargs.get('key')], on=merge_columns, how='outer')
            result_df.columns = merge_columns + ['Ground Truth', 'Simulation']
        elif not ground_truth_data is None and kwargs.get('key') in ground_truth_data:
            result_df = ground_truth_data[kwargs.get('key')].copy()
            result_df.rename(index=str,columns={"value":"Ground Truth"},inplace=True)
        elif not sim_data is None and kwargs.get('key') in sim_data:
            result_df = sim_data[kwargs.get('key')].copy()
            result_df.rename(index=str,columns={"value":"Simulation"},inplace=True)
        else:
            return None

        return result_df


def convert_dict_Series(ground_truth_data=None, sim_data=None, **kwargs):

    if kwargs.get('key'):

        both = True
        if not sim_data is None and kwargs.get('key') in sim_data:
            sim_data= sim_data[kwargs.get('key')]
            result_df = pd.DataFrame(sim_data).copy()
            result_df.rename(index=str,columns={"value":"Simulation"},inplace=True)
        else:
            both = False

        if not ground_truth_data is None and kwargs.get('key') in ground_truth_data:
            ground_truth_data=ground_truth_data[kwargs.get('key')]
            result_df = pd.DataFrame(ground_truth_data).copy()
            result_df.rename(index=str,columns={"value":"Ground Truth"},inplace=True)
        else:
            both = False

        if both:
            result_df = pd.concat([ground_truth_data.reset_index(drop=True),sim_data.reset_index(drop=True)], axis=1)
            result_df.columns = [ 'Ground Truth', 'Simulation']

        return result_df

def convert_dict_array(ground_truth_data=None, sim_data=None, **kwargs):

    if kwargs.get('key'):

        both = True

        if not sim_data is None:
            sim_data = pd.Series(sim_data[kwargs.get('key')])
            result_df = sim_data.copy()
            result_df.rename(index=str,columns={"value":"Simulation"},inplace=True)
        else:
            both = False

        if not ground_truth_data is None:
            ground_truth_data = pd.Series(ground_truth_data[kwargs.get('key')])
            result_df = ground_truth_data.copy()
            result_df.rename(index=str,columns={"value":"Simulation"},inplace=True)
        else:
            both = False

 
        if both:
            result_df = pd.concat([ground_truth_data,sim_data], axis=1)
            result_df.columns = [ 'Ground Truth', 'Simulation']

        return result_df
