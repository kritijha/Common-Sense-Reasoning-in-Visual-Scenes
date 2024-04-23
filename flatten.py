import os
import pickle
from tqdm import tqdm

# Define the source directories (slow and fast)
slow_dir = 'path/to/slow/directory'
fast_dir = 'path/to/fast/directory'

# Define the target directory
target_dir = 'path/to/target/directory'

# Create the target directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Get the list of files in the slow and fast directories
slow_files = [filename for filename in os.listdir(slow_dir) if filename.endswith('.pkl')]
fast_files = [filename for filename in os.listdir(fast_dir) if filename.endswith('.pkl')]

# Loop through all files in the slow directory with a progress bar
for slow_filename in tqdm(slow_files, desc="Processing files"):
    # Check if the file exists in the fast directory
    fast_filename = slow_filename.split('_')[0] + '_fast.pkl'
    if fast_filename in fast_files:
        # Read the pickle files from the slow and fast directories
        # slow_filename = filename.split('_')[0] + '_slow.pkl'
        slow_filepath = os.path.join(slow_dir, slow_filename)
        fast_filepath = os.path.join(fast_dir, fast_filename)
        try:
            with open(slow_filepath, 'rb') as f:
                slow_data = pickle.load(f)
        except EOFError:
            print(f"Error loading {slow_filepath}: EOFError")
            continue

        try:
            with open(fast_filepath, 'rb') as f:
                fast_data = pickle.load(f)
        except EOFError:
            print(f"Error loading {fast_filepath}: EOFError")
            continue
                
        # Concatenate the data
        concatenated = slow_data + fast_data
        
        # Check if the file already exists in the target directory
        filename = fast_filename.split('_')[0] + '_concat.pkl'
        target_filepath = os.path.join(target_dir, filename)
        if not os.path.exists(target_filepath):
            # Write the concatenated data to the target directory
            with open(target_filepath, 'wb') as f:
                pickle.dump(concatenated, f)
        else:
            print(f"Skipping {filename} as it already exists in the target directory")
    else:
        print(f"Skipping {filename} as it doesn't exist in the fast directory")