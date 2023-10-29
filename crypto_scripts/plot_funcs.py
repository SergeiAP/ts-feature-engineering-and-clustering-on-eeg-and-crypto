import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from tqdm.autonotebook import tqdm

from tslearn.clustering import TimeSeriesKMeans, silhouette_score


def plot_results(predicted: pd.Series,
                 representation: np.ndarray,
                 highlited: str = "BTC"):
    """tSNE plot"""
    idx = predicted.index.get_loc(highlited)
    coordinates = representation[idx]
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    scatter = ax.scatter(representation[:, 0], representation[:, 1], c=predicted)
    ax.vlines(coordinates[0], representation[:, 1].min(), coordinates[1],
              linestyle='dashed')
    ax.hlines(coordinates[1], representation[:, 0].min(), coordinates[0],
              linestyle='dashed')
    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper right", title="Classes")
    ax.add_artist(legend1)
    
    ax.set_title("Predicted clusters")
    plt.legend()
    plt.show()


def plot_cluster_tickers(current_cluster: pd.DataFrame):
    """Plot examples for specific cluster"""
    fig, ax = plt.subplots(2, 4, figsize=(15, 12))
    fig.autofmt_xdate(rotation=45)
    ax = ax.reshape(-1)

    for index, (_, row) in enumerate(current_cluster[:8].iterrows()):
        ax[index].plot(row[:-1])
        # ax[index].set_title(f"{row.shortName}\n{row.sector}")
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def plot_and_get_pca(df: pd.DataFrame,
                     seed: int,
                     is_plot: bool = False,
                     explained_tresh: float = 0.95) -> np.ndarray:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        seed (int): _description_
        is_plot (bool, optional): _description_. Defaults to False.
        explained_tresh (float, optional): _description_. Defaults to 0.95.

    Returns:
        np.ndarray: _description_
    """
    pca = PCA(random_state=seed)
    pca.fit(df)

    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    components_threshold = np.argwhere(
        explained_variance > explained_tresh).reshape(-1)[0]
    
    pca = PCA(n_components=components_threshold)
    pca_transformed = pca.fit_transform(df)
    print("Explained variance of 2 components", 
          np.sum(pca.explained_variance_ratio_[:2]))
    if is_plot:
        plt.vlines(components_threshold, explained_variance.min(),
                   explained_tresh, linestyle='dashed')
        plt.plot(explained_variance)
        plt.show()
    
    return pca_transformed


def plot_elbow(df: pd.DataFrame,
               seed: int,
               clusters: int = 15,
               n_init: int = 5,
               max_iter: int = 10,
               metric: str = "euclidean") -> None:
    """Plot elbow plot to determine num of clusters"""
    # Sum of squared distance from the item to cluster center 
    # weighted by weights (if exists)
    distortions = []
    silhouette = []
    K = range(2, clusters) # num of clusters
    for k in tqdm(K):
        kmeanModel = TimeSeriesKMeans(n_clusters=k,
                                      metric=metric,
                                      n_jobs=6,
                                      max_iter=max_iter,
                                      n_init=n_init,
                                      random_state=seed)
        kmeanModel.fit(df)
        distortions.append(kmeanModel.inertia_)
        silhouette.append(silhouette_score(df, kmeanModel.labels_, metric=metric))
        # silhouette_score calculate how 'clear' clusters
        
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(K, distortions, 'b-')
    ax2.plot(K, silhouette, 'r-')

    ax1.set_xlabel('# clusters')
    ax1.set_ylabel('Distortion', color='b')
    ax2.set_ylabel('Silhouette', color='r')

    plt.show()
