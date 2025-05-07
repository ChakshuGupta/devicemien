import numpy as np
import os
import pandas as pd
import pickle
import random
import sys
# import torch

from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

from extract_features import extract_flows, get_flow_windows
from objects.kmeans_tf import KMeansTF
# from objects.device_classifier import DeviceClassifier
from train_test import train_lstm_ae
from util import get_pcap_list, load_device_file


def full_reproducibility(seed=123):
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


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
    
    full_reproducibility()
    
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
    y_prob_all = []
    # Declare the stratified k fold object
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=1234)
    idx = 0
    # Loop through the different folds
    for train_index, test_index in skf.split(dataset_x, dataset_y):
        # split the dataset into train and test dataset using the indices
        x_train = dataset_x[train_index]
        y_train = dataset_y[train_index]
        x_test = dataset_x[test_index]
        y_test = dataset_y[test_index]

        model = train_lstm_ae(x_train)

        x_train_encoded = model.reconstruct(x_train)
        x_test_encoded = model.reconstruct(x_test)
        print(x_train_encoded.shape, x_test_encoded.shape)

        # x_train_encoded = x_train_encoded.detach()

        # Convert latent test vectors to numpy for clustering prediction
        print(x_test_encoded.shape)
        # x_test_encoded = x_test_encoded.detach()
        print(x_test_encoded.shape)

        print(np.unique(dataset_y))

        # classifier = DeviceClassifier(unique_label = len(np.unique(dataset_y)), max_k=3*len(np.unique(dataset_y)))
        classifier =  KMeansTF(n_clusters=len(np.unique(dataset_y))+1)
        classifier.fit(x_train_encoded, y_train)
        # print(classifier.n_clusters)
        # classifier.fit(x_train_encoded, y_train)
        # print(classifier.n_clusters)

        # y_preds, y_probs = classifier.predict(x_test_encoded)
        # # print(list(zip(y_test, y_preds, y_probs)))

        
        y_preds, y_probs = classifier.predict(x_test_encoded)
        print(list(zip(y_preds, y_probs)))
        y_true_all.extend(y_test)
        y_pred_all.extend(y_preds)
        # y_prob_all.extend(y_probs)
        # Plot results
        # classifier.plot_clusters(x_train_encoded)


    report = classification_report(y_true_all, y_pred_all, zero_division=0.0)
    print(report)