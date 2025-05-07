import os
import random

# # Define directories
# real_dir = r'/data/ssd2/shakeel-workspace/DOWNLOAD/SiW-Mv2_preprocessed/Live/'
# spoof_dir = r'/data/ssd2/shakeel-workspace/DOWNLOAD/SiW-Mv2_preprocessed/Spoof_root/'

# Reference files
train_spoof_ref = r'/data/ssd2/shakeel-workspace/DOWNLOAD/SiW-Mv2_preprocessed/trainlist_all.txt'
train_live_ref = r'/data/ssd2/shakeel-workspace/DOWNLOAD/SiW-Mv2_preprocessed/trainlist_live.txt'

# Output train and val files
train_spoof_out = r'protocol_1/train_spoof.txt'
val_spoof_out = r'protocol_1/val_spoof.txt'
train_live_out = r'protocol_1/train_live.txt'
val_live_out = r'protocol_1/val_live.txt'

# Split ratio
val_ratio = 0.2

def read_ref_file(file_path):
    """Reads folder names from a reference file."""
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def split_data(folders, val_ratio):
    """Splits the folder names into train and val sets."""
    random.shuffle(folders)
    split_idx = int(len(folders) * (1 - val_ratio))
    return folders[:split_idx], folders[split_idx:]

def write_to_file(file_path, data):
    """Writes folder names to a file."""
    with open(file_path, 'w') as file:
        for folder in data:
            file.write(folder + '\n')

# Read folder names
train_spoof_folders = read_ref_file(train_spoof_ref)
train_live_folders = read_ref_file(train_live_ref)

# Split into train and validation sets
train_spoof, val_spoof = split_data(train_spoof_folders, val_ratio)
train_live, val_live = split_data(train_live_folders, val_ratio)

# Write train and validation sets to files
write_to_file(train_spoof_out, train_spoof)
write_to_file(val_spoof_out, val_spoof)
write_to_file(train_live_out, train_live)
write_to_file(val_live_out, val_live)

print("Data splitting completed!")
