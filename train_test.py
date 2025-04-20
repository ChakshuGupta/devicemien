import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch

from objects.lstm import StackedAutoencoder
# from objects.stacked_lstm_ae import StackedAutoencoder
from util import convert_to_tensor


def lstmae_encode(x_train, x_test):
    """
    """

    # Convert the dataframes to tensors
    x_train = convert_to_tensor(x_train)
    x_test = convert_to_tensor(x_test)


    print(x_train.shape)

    n_seq, seq_len, n_features = x_train.shape
    
    model = StackedAutoencoder(seq_len, n_features)

    encoded_train = model.encode(x_train)
    encoded_test = model.encode(x_test)    

    return encoded_train, encoded_test


# Function to perform clustering and get distribution
def cluster_and_get_distribution(encoded_input, num_clusters=10):
    """
    """
    print(type(encoded_input))
    # Perform K-Means clustering on the latent vectors
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    encoded_input = encoded_input.detach().numpy()
    kmeans.fit(encoded_input)

    # Get the cluster labels for each sample
    labels = kmeans.labels_

    # Calculate the distribution (percentage) of samples per cluster
    unique, counts = np.unique(labels, return_counts=True)
    cluster_distribution = dict(zip(unique, counts))
    total_samples = len(labels)
    
    print(f"Cluster Distribution (over {total_samples} samples):")
    for cluster, count in cluster_distribution.items():
        print(f"Cluster {cluster}: {count} samples ({(count / total_samples) * 100:.2f}%)")

    # # Optional: Visualize the clusters if latent_dim is 2
    # if encoded_input.shape[1] == 2:  # If latent vector has 2 dimensions
    #     plt.scatter(encoded_input[:, 0], encoded_input[:, 1], c=labels, cmap='viridis')
    #     plt.title('Clustered Latent Representations')
    #     plt.xlabel('Latent Dim 1')
    #     plt.ylabel('Latent Dim 2')
    #     plt.show()

    return labels, cluster_distribution, kmeans