import pandas as pd
from preprocessing.config import settings
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.cluster import KMeans,SpectralClustering
from sklearn.metrics import silhouette_score
from modelling import k_means

def biplot(df, pca_coef, pca_score, show_plot=True):
    font_title = {'family': 'serif',
                  'color': 'darkred',
                  'weight': 'normal',
                  'size': 16, }

    if 'z' in df.columns:
        df.drop(['z'],axis=1, inplace=True)

    #print(df)
    xs = pca_score[:, 0]
    ys = pca_score[:, 1]
    scale_x = 1.0 / (xs.max() - xs.min())
    scale_y = 1.0 / (ys.max() - ys.min())

    fig, ax = plt.subplots()
    ax.scatter(xs * scale_x, ys * scale_y, s=8, c='darkblue')


    col = list(df.columns)
    n = pca_coef.shape[0]
    for i in range(n):
        ax.arrow(0, 0, pca_coef[i, 0] * 0.8, pca_coef[i, 1] * 0.8, color='crimson', alpha=2)
        ax.text(pca_coef[i, 0] * 0.87, pca_coef[i, 1] * 0.87, col[i], color='black',
                ha='center', va='center', weight='bold')


    k = df.reset_index()
    names = k['Player'].str.split(" ", expand=True)
    names_index = names[0].str[0] + '.' + names[1]
    n = pd.DataFrame(data={"xs": xs, "ys": ys}, index=names_index)
    right = n.sort_values(by=["xs", "ys"], ascending=False)[0:2]
    up = n.sort_values(by=["ys", "xs"], ascending=False)[2:4]
    left = n.sort_values(by=["xs", "ys"], ascending=True)[0:1]

    for i, txt in enumerate(right.index):
        ax.annotate(txt, (right['xs'][i] * scale_x, right['ys'][i] * scale_y), ha='center')

    for i, txt in enumerate(left.index):
        ax.annotate(txt, (left['xs'][i] * scale_x, left['ys'][i] * scale_y))

    for i, txt in enumerate(up.index):
        ax.annotate(txt, (up['xs'][i] * scale_x, up['ys'][i] * scale_y))

    ax.set_xlabel("PC{}".format(1))
    ax.set_ylabel("PC{}".format(2))

    ax.grid(True)
    ax.set_xlim([-0.70, 0.65])
    ax.set_ylim([-0.55, 0.58])
    if show_plot:
        plt.show()

def choose_K_means(df, k=settings.MODELS.KMEANS_CLUSTERS):
    raw_df = df.reset_index()
    X = raw_df.drop(['z','Player'], axis=1)

    Sum_of_squared_distances = []
    K = range(1, k)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(X)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

def choose_K_means_sil(df, k=settings.MODELS.KMEANS_CLUSTERS):
    raw_df = df.reset_index()
    X = raw_df.drop(['z', 'Player'], axis=1)

    sil = []
    for k in range(2, k + 1):
        kmeans = KMeans(n_clusters=k).fit(X)
        labels = kmeans.labels_
        sil.append(silhouette_score(X, labels, metric='euclidean'))
    plt.plot(range(2, k + 1),sil)
    plt.xlabel('k')
    plt.ylabel('silhouette score')
    plt.title('Silhouette score by K')
    plt.show()

def spectral_Kmeans(df, clusters=settings.MODELS.KMEANS_CLUSTERS):
    raw_df = df.reset_index()
    X = raw_df.drop(['z', 'Player'], axis=1)

    X = X.to_numpy()
    model = SpectralClustering(n_clusters=clusters, affinity='nearest_neighbors',
                               assign_labels='kmeans')
    labels = model.fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    plt.show()

def Kmeans_plot(df, pca_score,centers,groups):
    # k = k_means(df,clusters)
    # centers = k[1]
    # groups = k[2]
    xs = pca_score[:, 0]
    ys = pca_score[:, 1]
    df = pd.DataFrame(dict(x=xs, y=ys, label=groups))
    groups = df.groupby(df.label)

    fig, ax = plt.subplots()

    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', label=name)

    ax.legend()
    centers = centers
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

    plt.show()