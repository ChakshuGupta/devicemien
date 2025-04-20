import numpy as np
import os
import torch

def load_device_file(device_file):
   """
   Load the mapping between devices and mac addresses
   """
   file_data = open(device_file, "r")
   device_mac_map = {}
   
   for line in file_data:
       if line.strip() == "":
           continue
       device = line.split(",")[0]
       mac = line.split(",")[1]
       device_mac_map[mac.strip()] = device.strip()

   return device_mac_map


def get_pcap_list(dataset_dir):
    """
    Get the list of pcap files in the directory
    """
    pcap_files = []

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".pcap") or file.endswith(".pcapng"):
                pcap_files.append(os.path.join(root, file))
    pcap_files.sort()
    return pcap_files


def convert_to_tensor(features):
    """
    """
    # Convert features to tensor format
    tensor_features = torch.from_numpy(features.astype(np.float32))

    return tensor_features