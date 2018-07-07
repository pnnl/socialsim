import pandas as pd


def load_data():
    
    path = '/Users/grac833/Documents/Projects/SocialSim/temp/infrastructure/tira/services/GithubMetricServices'

    dfs = []
    for i in range(1,3):
        i = str(i)
        if len(i) == 1:
            i = '0' + i
        df = pd.read_csv(path + '/leidosData/weekly_data_2017-07-' + str(i) + ' 00:00:00.csv')
        dfs.append(df)
        df = pd.read_csv(path + '/leidosData/weekly_data_2017-08-' + str(i) + ' 00:00:00.csv')
        dfs.append(df)
    gt = pd.concat(dfs)

    dfs = []
    for i in range(1, 3):
        i = str(i)
        if len(i) == 1:
            i = '0' + i
        df = pd.read_csv(path + '/leidosData/weekly_data_2017-07-' + str(i) + ' 00:00:00.csv')
        dfs.append(df)
        df = pd.read_csv(path + '/leidosData/weekly_data_2017-08-' + str(i) + ' 00:00:00.csv')
        dfs.append(df)
    sim1 = pd.concat(dfs)

    gt = gt.drop("_id", axis=1)
    sim1 = sim1.drop("_id", axis=1)

    print(sim1)

    return gt,sim1


if __name__ == "__main__":

    load_data()
