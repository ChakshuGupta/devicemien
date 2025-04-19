import json
import numpy as np
import os
import pandas as pd
import pickle
import sys

from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

from extract_features import extract_flows, get_flow_windows
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


    if os.path.isdir(dataset_path):
        pcap_list = get_pcap_list(dataset_path)
        flows = extract_flows(pcap_list)
        features = get_flow_windows(flows, mac_addrs)
        print(features)