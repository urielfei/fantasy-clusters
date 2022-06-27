import pandas as pd
from sklearn.preprocessing import StandardScaler
from preprocessing.config import settings
from nba_api.stats.endpoints import leaguedashplayerstats

class nbaData:
    def __init__(self,my_df):
        self.raw_df = my_df
        self.df = None
        self.df_scaled = None
        self.names = None

    def get_data(self, mode='per_game',scaled=False):
        features = settings.NAMES.FEATURES
        features_dict = settings.NAMES.FEATURES_DICT

        df = self.raw_df[features].rename(features_dict, axis='columns')
        df = df[df['GAMES'] >= settings.PARAMS.MIN_GAMES_PLAYER_PLAYED]

        names = df["Player"]
        x = df.set_index('Player')

        if mode == 'per_game':
            features_to_divide = ['3PTS', 'PTS', 'REB', 'STL', 'BLK', 'TOV']
            x[features_to_divide] = x[features_to_divide].div(x['GAMES'], axis=0)

        x['TOV'] = x['TOV'] * -1
        df = x.drop('GAMES', axis=1)

        if scaled:
            sc = StandardScaler()
            sc.fit(df)
            x_scaled = sc.transform(df)
            x = pd.DataFrame(data=x_scaled, columns=features[2:11], index=names)
            x['z'] = x.sum(axis=1)
            df = x.sort_values(by='z', ascending=False)

        self.df = df[0:settings.PARAMS.N_PLAYERS_FINAL_DF]
        self.names = self.df.index
        return self.df

#raw = leaguedashplayerstats.LeagueDashPlayerStats(season='2019-20').get_data_frames()[0]
# raw = pd.read_csv('raw_data.csv')
# obj = nbaData(raw)
# df = obj.get_data(scaled=True)
#df.to_csv('data_scaled.csv')