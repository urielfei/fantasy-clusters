import pandas as pd
import numpy as np
from preprocessing.config import settings

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from nba_api.stats.endpoints import leaguedashplayerstats



def k_means(df, clusters=settings.MODELS.KMEANS_CLUSTERS):
    raw_df = df
    raw_df = raw_df.reset_index()
    X = raw_df.drop('z', axis=1)
    X.set_index('Player',inplace=True)

    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    df_y_kmeans = pd.DataFrame(data=y_kmeans, columns=['Group'])
    df_groups = pd.concat([raw_df, df_y_kmeans], axis=1).set_index('Player')
    centers = kmeans.cluster_centers_
    groups = y_kmeans
    return df_groups,centers,groups

def Kmeans_results(df,clusters=settings.MODELS.KMEANS_CLUSTERS,data='reg'):
    if data == 'reg':
        data_k_means = k_means(df,clusters)[0]
        data_k_means = data_k_means[['z','Group']]
        groups = data_k_means.groupby(['Group'])

        for key, group in groups:
            print(key)
            print(groups.get_group(key).head())
    print('### Mean Z by Group')
    print(data_k_means.groupby(['Group']).mean())

def pca(df):
    X = df
    X = X.drop(['z'],axis=1)
    if 'Player' in X.columns:
        X.set_index('Player',inplace=True)


    pca = PCA(n_components= settings.MODELS.PCA_N_COMPONENTS)
    pca_x = pca.fit_transform(X)

    pca_coef = np.transpose(pca.components_[0:2, :])
    pca_score = pca_x[:, 0:2]
    return pca_coef, pca_score


# raw = pd.read_csv('data_scaled.csv')
# k = k_means(raw,3)
# print(k)
#
# p = pca(raw)
# print(p)
# obj = Nba_data(raw)
# df = obj.get_data(scaled=True)
