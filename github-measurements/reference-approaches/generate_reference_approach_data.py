import pandas as pd
import datetime
import numpy as np
import glob


def ingest_historical_data(csv_file):

    """
    Read data from csv file
    """

    print('reading data...')
    df = pd.read_csv(csv_file)
    df.columns = ['created_at','type','actor_login_h','repo_name_h','payload_action','payload_pull_request_merged']

    print('to datetime..')
    df['created_at'] = pd.to_datetime(df['created_at'])

    print('sorting...')
    df = df.sort_values('created_at')

    return df

def subset_data(df,start,end):

    """
    Return temporal data subset based on start and end dates
    """

    print('subsetting...')
    df = df[ (df['created_at'] >= start) & (df['created_at'] <= end) ]

    return(df)

def shift_data(df,shift, end):

    """
    Shift data based on fixed offset (shift) and subset based on upper limit (end)
    """

    print('shifting...')
    df['created_at'] += shift
    df = df[df['created_at'] <= end]

    return df


def sample_data(df,start,end,proportional=True):

    """
    Sample data either uniformly (proportional=False) or proporationally (proportional=True) to fill test period from start to end
    """

    print('inter-event times...')

    df['inter_event_times'] = df['created_at'] - df['created_at'].shift()
    inter_event_times = df['inter_event_times'].dropna()

    max_time = df['created_at'].min()
    multiplier=( (pd.to_datetime(end) - pd.to_datetime(start)) / df['inter_event_times'].mean() ) / float(len(df.index))

    #repeat until enough data is sampled to fill the test period
    while max_time < pd.to_datetime(end):

        if proportional:
            sample = pd.DataFrame(df['inter_event_times'].dropna().sample(int(multiplier*len(df.index)),replace=True))
            sampled_inter_event_times = sample.cumsum()
        else:
            sample = pd.DataFrame(np.random.uniform(np.min(inter_event_times.dt.total_seconds()),1.0,int(multiplier*len(df.index))))[0].round(0)
            sample = pd.to_timedelta(sample,unit='s')
            sampled_inter_event_times = pd.DataFrame(sample).cumsum()
        
        event_times = (pd.to_datetime(start) + sampled_inter_event_times)
        max_time = pd.to_datetime(event_times.max().values[0])
        multiplier*=1.5

    event_times = event_times[(event_times < pd.to_datetime(end)).values]

    if proportional:
        users = df['actor_login_h']
        repos = df['repo_name_h']
        events = df['type']
    else:
        users = pd.Series(df['actor_login_h'].unique())
        repos = pd.Series(df['repo_name_h'].unique())
        events = pd.Series(df['type'].unique())


    users = users.sample(len(event_times),replace=True).values
    repos = repos.sample(len(event_times),replace=True).values
    events = events.sample(len(event_times),replace=True).values

    df_out = pd.DataFrame({'time':event_times.values.flatten(),
                           'event':events,
                           'user':users,
                           'repo':repos})
    
    if proportional:
        pr_action = df[df['type'] == 'PullRequestEvent']['payload_action']
        pr_merged = df[df['type'] == 'PullRequestEvent']['payload_pull_request_merged']
        iss_action = df[df['type'] == 'IssuesEvent']['payload_action']
    else:
        pr_action = df[df['type'] == 'PullRequestEvent']['payload_action'].unique()
        pr_merged = df[df['type'] == 'PullRequestEvent']['payload_pull_request_merged'].unique()
        iss_action = df[df['type'] == 'IssuesEvent']['payload_action'].unique()

    pull_requests = df_out[df_out['event'] == 'PullRequestEvent']
    pull_requests['payload_action'] = pd.Series(pr_action).sample(len(pull_requests.index),
                                                                  replace=True).values
    pull_requests['payload_pull_request_merged'] = pd.Series(pr_merged).sample(len(pull_requests.index),
                                                                               replace=True).values


    issues = df_out[df_out['event'] == 'IssuesEvent']
    issues['payload_action'] = pd.Series(iss_action).sample(len(issues.index),replace=True).values
    
    df_out = df_out[~df_out['event'].isin(['IssuesEvent','PullRequestEvent'])]
    df_out = pd.concat([df_out,pull_requests,issues])
    df_out = df_out.sort_values('time')

    df_out = df_out[['time','event','user','repo','payload_action','payload_pull_request_merged']]

    return df_out


def create_shifted_reference(csv_file, test_start_date='2018-02-01', test_end_date='2018-02-28',
                             historical_start_date='2017-08-01',historical_end_date='2017-08-31'):


    """
    Create shifted reference from historical data in csv_file using data ranging from historical_start_date
    to historical_end_date to generate new shifted data ranging from test_start_date to test_end_date.
    """


    df = ingest_historical_data(csv_file)


    test_delta_t = np.datetime64(test_end_date) - np.datetime64(test_start_date)
    historical_delta_t = np.datetime64(historical_end_date) - np.datetime64(historical_start_date)
    if historical_delta_t > test_delta_t:
        df = subset_data(df,historical_start_date,historical_end_date)
    else:
        print('Not enough historical data to create shifted reference approach')
        return None

    shifted_df = shift_data(df,np.datetime64(test_start_date) - np.datetime64(historical_start_date),np.datetime64(test_end_date))    
    shifted_df = subset_data(shifted_df,test_start_date,test_end_date)

    return shifted_df


def create_sampled_reference(csv_file, test_start_date='2018-02-01', test_end_date='2018-02-28',
                             historical_start_date='2017-08-01',historical_end_date='2017-08-31',
                             proportional=True):

    """
    Create sampled reference from historical data in csv_file using data ranging from historical_start_date
    to historical_end_date to generate new sampled data ranging from test_start_date to test_end_date.
    If proportional is True, the sampling will be proportional to the observed frequencies in the
    historical data.  Otherwise, sampling will be uniform.
    """

    df = ingest_historical_data(csv_file)

    df = subset_data(df,historical_start_date,historical_end_date)
   
    sampled_df = sample_data(df,test_start_date, test_end_date,proportional)

    return sampled_df

    
def main():

    fn = 'august_2017.csv'

    shifted_reference = create_shifted_reference(fn,test_end_date='2018-02-05')
    print('shifted reference')
    print(shifted_reference)

    sampled_reference_uniform = create_sampled_reference(fn,proportional=False,test_end_date='2018-02-05')
    print('sampled reference uniform')
    print(sampled_reference_uniform)

    sampled_reference_proportional = create_sampled_reference(fn,proportional=True,test_end_date='2018-02-05')
    print('sampled reference proportional')
    print(sampled_reference_proportional)
    

if __name__ == "__main__":
    main()
