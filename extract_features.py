import ipaddress
import numpy as np
import pandas as pd

from scapy.all import *

from objects.feature_vector import FeatureVector

SEQ_LEN = 10         # packets per sample

def get_direction(packet):
    """
    Get the direction of the traffic: inbound (0) or outbound (1)
    """
    if "IPv6" in packet:
        ip_layer = packet[IPv6]

    elif "IP" in packet:
        ip_layer = packet[IP]
    else:
        return 0

    src = ipaddress.ip_address(ip_layer.src)   
    dst = ipaddress.ip_address(ip_layer.dst)

    if src.is_global:
        return 0
    elif dst.is_global:
        return 1
    else:
        return 0


def extract_flows(list_pcaps):
    """
    Parse the pcap file and extracts the features from the traffic.
    """

    flows = dict()
    
    for pcap_file in list_pcaps:
        print("Reading file: ", pcap_file)
        packets = rdpcap(pcap_file)

        if not packets:
            raise ValueError("No packets found in the pcap file: ", pcap_file)
        
        for packet in packets:
             
            if 'IP' not in packet:
                continue

            if 'TCP' not in packet:
                continue
            
            flow_key = (packet['IP'].src, packet['IP'].dst, packet['TCP'].sport, packet['TCP'].dport, packet['IP'].proto)
            reverse_key = (packet['IP'].dst, packet['IP'].src, packet['TCP'].dport, packet['TCP'].sport, packet['IP'].proto)
            key = flow_key
            
            # Check if the flow exists in the flows dictionary
            if flow_key not in flows and reverse_key not in flows:
                flows[key] = list()
            elif flow_key in flows:
                key = flow_key              
            else:
                key = reverse_key
            
            flows[key].append(packet)
    
    return flows


def extract_features(packet, flow_start):
    """
    Extract the features from the TCP packets

    """
    feature_vector = FeatureVector()

    feature_vector.time_delta = packet.time - flow_start
    feature_vector.direction = get_direction(packet)
    feature_vector.sport = packet["TCP"].sport
    feature_vector.dport = packet["TCP"].dport
    feature_vector.length = len(packet["TCP"])
    feature_vector.flags = int(packet["TCP"].flags)
    feature_vector.window = packet["TCP"].window
    feature_vector.ttl = packet["IP"].ttl

    print(feature_vector.__dict__)
    return feature_vector.__dict__    


def get_flow_windows(flows, mac_addrs):
    # Create flow-based samples: N-packet windows
    dataset = list()

    for flow_pkts in flows.values():
        if len(flow_pkts) < SEQ_LEN:
            continue

        flow_pkts = sorted(flow_pkts, key=lambda x: x.time)
        flow_start = flow_pkts[0].time

        if flow_pkts[0]["Ether"].src in mac_addrs:
            device = mac_addrs[flow_pkts[0]["Ether"].src]
        else:
            device = mac_addrs[flow_pkts[0]["Ether"].dst]

        for i in range(len(flow_pkts) - SEQ_LEN + 1):
            window = flow_pkts[i: i + SEQ_LEN]
            try:
                features = [extract_features(packet, flow_start) for packet in window]
                df_features = pd.DataFrame.from_dict(features)
                dataset.append((df_features, device))
                print(device)
            except Exception:
                print("ERROR! Could not extract the features!")
                continue

    print(f"Generated {len(dataset)} flow samples")
    return np.array(dataset, dtype=object)  # shape: [samples, seq_len, features]