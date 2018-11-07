import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set(style="whitegrid")


def histogram(df, xlabel, ylabel, title, **kwargs):
    n_bins = 100

    if 'Simulation' in df.columns and 'Ground Truth' in df.columns:

        gold_data = df.dropna(subset=["Ground Truth"])["Ground Truth"]
        test_data = df.dropna(subset=["Simulation"])["Simulation"]
       
        data = np.concatenate([gold_data, test_data])
    
    elif 'Simulation' in df.columns or 'Ground Truth' in df.columns:

        if 'Simulation' in df.columns:
            data = df.dropna(subset=["Simulation"])['Simulation']
            test_data = data.copy()
        else: 
            data = df.dropna(subset=["Ground Truth"])['Ground Truth']
            gold_data = data.copy()
    else:
        return None

    _,bins = np.histogram(data,bins='doane')
    #bins = np.linspace(data.min(), data.max(), n_bins)

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    if 'Ground Truth' in df.columns:
        ax.hist(gold_data, bins, log=True, label='Ground Truth', alpha=0.7, color='green')
    if 'Simulation' in df.columns:
        ax.hist(test_data, bins, log=True, label='Simulation', alpha=.7, color='red')

    ax.set(xlabel=xlabel)
    ax.set(ylabel=ylabel)
    ax.set(title=title)
    ax.legend(loc='best')

    plt.tight_layout()
    return fig


def scatter(df, xlabel, ylabel, title, **kwargs):

    if 'Ground Truth' in df.columns and 'Simulation' in df.columns:
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        sns.scatterplot(x="Ground Truth", y="Simulation", data=df, ax=ax, alpha=0.7)
        ax.set(xlabel=xlabel)
        ax.set(ylabel=ylabel)
        ax.set(title=title)
        plt.tight_layout()
        return fig
    else:
        return None


def bar(df, xlabel, ylabel, title, **kwargs):

    palette = set_palette(df)

    df.fillna(0, inplace=True)

    df = df.melt(df.columns[0], var_name='type', value_name='vals')

    fig, ax = plt.subplots(1, 1, figsize=(15, 7))
    sns.barplot(x=df.columns[0], y='vals', hue='type', data=df, ax=ax, palette=palette, alpha=0.7)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.set(xlabel=xlabel)
    ax.set(ylabel=ylabel)
    ax.legend(loc='best')
    ax.set(title=title)
    plt.tight_layout()
    return fig


def set_palette(df):

    if 'Ground Truth' in df.columns and 'Simulation' in df.columns:
        palette = ['green','red']
    elif 'Ground Truth' in df.columns:
        palette = ['green']
    else:
        palette = ['red']
    
    return palette

def time_series(df, xlabel, ylabel, title, **kwargs):
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    palette = set_palette(df)

    df = df.melt(id_vars = [c for c in df.columns if c not in ['Ground Truth','Simulation']], var_name='type', value_name='vals').sort_values('type')

    df.dropna(inplace=True)
    sns.lineplot(x=df.columns[0], y='vals', hue='type', data=df, ax=ax, marker='o', palette=palette, alpha=0.7)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc='best', handles=handles[1:], labels=labels[1:])

    ax.set(xlabel=xlabel)
    ax.set(ylabel=ylabel)
    ax.set(title=title)
    plt.tight_layout()
    
    return fig
    


def multi_time_series(df, xlabel, ylabel, title, **kwargs):

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    if 'time' in df.columns:
        time_col = 'time'
    elif 'date' in df.columns:
        time_col = 'date'
    elif 'weekday' in df.columns:
        day_map = {'Monday':1,
               'Tuesday':2,
               'Wednesday':3,
               'Thursday':4,
               'Friday':5,
               'Saturday':6,
               'Sunday':7}
        df['weekday_int'] = df['weekday'].map(day_map)
        df = df.sort_values('weekday_int')
        time_col = 'weekday_int'

    if 'Ground Truth' in df.columns and 'Simulation' in df.columns:
        value_vars = ['Ground Truth', 'Simulation']
    elif 'Ground Truth' in df.columns:
        value_vars = ['Ground Truth']
    else:
        value_vars = ['Simulation']

    df = pd.melt(df, id_vars=[c for c in df.columns if c not in value_vars], value_vars=value_vars, var_name='type').fillna(0)

    sns.lineplot(x=time_col, y='value', hue=[c for c in df.columns if c not in ['Ground Truth', 'Simulation',time_col]][0], style='type', 
                 data=df, ax=ax, marker='o', alpha=0.7,
                 palette='bright')

    if time_col == 'weekday_int':
        ax.set(xticklabels=df['weekday'].unique())

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc='best', handles=handles[1:], labels=labels[1:])
    ax.set(xlabel=xlabel)
    ax.set(ylabel=ylabel)
    ax.set(title=title)
    plt.tight_layout()
    return fig
    

def save_charts(fig, loc):
    fig.savefig(loc)
    plt.close(fig)


def show_charts():
    plt.show()

def chart_factory(chart_name):
    charts_mapping = {
        'bar': bar,
        'hist': histogram,
        'time_series': time_series,
        'scatter': scatter,
        'multi_time_series':multi_time_series
    }

    return charts_mapping.get(chart_name, None)
