import numpy as np
import os
import pandas as pd
import pickle
import sys

from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

from extract_features import extract_flows, get_flow_windows
from train_test import lstmae_encode, cluster_and_get_distribution
from util import get_pcap_list, load_device_file


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("ERROR! THe script requires the path to the dataset.")
        exit(1)

    dataset_path = sys.argv[1]
    
    if not os.path.isdir(dataset_path):
        print("ERROR! The given path is not a directory.")
        exit(1)
    
    device_file = os.path.join(dataset_path, "devices.txt")

    mac_addrs = load_device_file(device_file)
    list_devices = list(mac_addrs.values())

    features_file = "features.pickle"
    labels_file = "labels.pickle"

    if not os.path.isdir(dataset_path):
        print("ERROR! Dataset path doesn't exist!")
    
    if os.path.isfile(features_file):
        dataset_x = pickle.load(open(features_file, "rb"))
        dataset_y = pickle.load(open(labels_file, "rb"))
    else:
        pcap_list = get_pcap_list(dataset_path)
        flows = extract_flows(pcap_list)
        dataset = get_flow_windows(flows, mac_addrs)
        print("Saving the extracted features into pickle files.")
        
        dataset_x, dataset_y = zip(*dataset)
        df_dataset_y = pd.DataFrame(dataset_y)
        # Save the dataframes to pickle files    
        pickle.dump(dataset_x, open(features_file, "wb"))
        pickle.dump(df_dataset_y, open(labels_file, "wb"))

    dataset_x = np.array(dataset_x, dtype=object)
    dataset_y = np.array(dataset_y, dtype=object)

    # Declare the lists
    y_true_all = []
    y_pred_all = []
    # Declare the stratified k fold object
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1111)
    idx = 0
    # Loop through the different folds
    for train_index, test_index in skf.split(dataset_x, dataset_y):
        # split the dataset into train and test dataset using the indices
        x_train = dataset_x[train_index]
        y_train = dataset_y[train_index]
        x_test = dataset_x[test_index]
        y_test = dataset_y[test_index]

        x_train_encoded, x_test_encoded = lstmae_encode(x_train, x_test)
        print(x_train_encoded, x_test_encoded)

        labels, dist, kmeans = cluster_and_get_distribution(x_test_encoded, len(list_devices))
        print(labels, dist)

        # Convert latent test vectors to numpy for clustering prediction
        x_test_encoded = x_test_encoded.detach().cpu().numpy()

        # Use the trained KMeans model to predict the clusters for the test data
        test_labels = kmeans.predict(x_test_encoded)

        # Print the cluster labels for the test data
        print("Test Data Cluster Labels:")
        print(test_labels)