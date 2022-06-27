import pandas as pd
import numpy as np
from preprocessing.config import settings
from data_transformations import nbaData
from modelling import k_means, Kmeans_results,pca
from plot import biplot,choose_K_means,choose_K_means_sil,spectral_Kmeans, Kmeans_plot
from sklearn.preprocessing import StandardScaler
from nba_api.stats.endpoints import leaguedashplayerstats

raw = leaguedashplayerstats.LeagueDashPlayerStats(season='2019-20').get_data_frames()[0]
nba_data_obj = nbaData(raw)
df = nba_data_obj.get_data(scaled=True)

k = k_means(df,3)
centers = k[1]
groups = k[2]

p = pca(df)
pca_coef = p[0]
pca_scores = p[1]

biplot(df,pca_coef,pca_scores)
choose_K_means(df,10)
choose_K_means_sil(df,10)
spectral_Kmeans(df,5)
Kmeans_results(df)
Kmeans_plot(df,pca_scores,centers,groups)