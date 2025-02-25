import os
import requests
import json
import yaml

# Directories for raw and processed data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_cmu_dataset")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "deepmimic_dataset")

# Processed files expected for our project
REQUIRED_PROCESSED_FILES = ["motion_data.json", "metadata.yaml"]

# URL for the raw AMC file (direct download, not a zip)
CMU_AMC_URL = "http://mocap.cs.cmu.edu/subjects/06/06_14.amc"

def check_processed_dataset():
    """
    Check if the processed dataset exists with all required files.
    """
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
        return False
    for filename in REQUIRED_PROCESSED_FILES:
        if not os.path.exists(os.path.join(PROCESSED_DATA_DIR, filename)):
            return False
    return True

def download_raw_amc():
    """
    Download the raw AMC file from the given URL and save it into RAW_DATA_DIR.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    filename = CMU_AMC_URL.split("/")[-1]
    file_path = os.path.join(RAW_DATA_DIR, filename)
    if os.path.exists(file_path):
        print("Raw AMC file already exists at", file_path)
        return True

    print("Downloading raw AMC file from:", CMU_AMC_URL)
    try:
        response = requests.get(CMU_AMC_URL)
        response.raise_for_status()
    except Exception as e:
        print("Error downloading raw AMC file:", e)
        return False

    with open(file_path, "wb") as f:
        f.write(response.content)
    print("Downloaded and saved raw AMC file to", file_path)
    return True

def convert_cmu_to_deepmimic_format():
    """
    Convert raw CMU AMC file into the processed format required by DeepMimic.
    This example processes the '06_14.amc' file and converts it into a JSON
    file containing frames of joint data, along with a simple YAML metadata file.
    """
    print("Converting raw CMU AMC data to DeepMimic format...")
    amc_file_path = os.path.join(RAW_DATA_DIR, "06_14.amc")
    if not os.path.exists(amc_file_path):
        print("Expected AMC file not found:", amc_file_path)
        return False

    try:
        with open(amc_file_path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        print("Error reading AMC file:", e)
        return False

    frames = []
    current_frame = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # If the line is a digit, it denotes a new frame.
        if line.isdigit():
            if current_frame is not None:
                frames.append(current_frame)
            current_frame = {"frame": int(line), "joints": {}}
        else:
            # Skip joint data until we've encountered the first frame.
            if current_frame is None:
                continue
            # Each subsequent line provides joint name and its values.
            parts = line.split()
            if len(parts) > 1:
                joint_name = parts[0]
                try:
                    values = list(map(float, parts[1:]))
                except ValueError:
                    values = []
                current_frame["joints"][joint_name] = values
    if current_frame is not None:
        frames.append(current_frame)

    # Create the motion data structure expected by our DeepMimic model.
    motion_data = {"frames": frames}

    # Ensure the processed data directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Write the motion data to a JSON file.
    motion_data_path = os.path.join(PROCESSED_DATA_DIR, "motion_data.json")
    with open(motion_data_path, "w") as f:
        json.dump(motion_data, f, indent=4)
    print("Motion data saved to", motion_data_path)

    # Create a minimal metadata file.
    metadata = {
        "dataset": "CMU Graphics Lab Motion Capture Database",
        "scenario": "06_14",
        "num_frames": len(frames),
        "description": "Converted AMC file from CMU dataset to DeepMimic format. Additional processing may be needed for full fidelity."
    }
    metadata_path = os.path.join(PROCESSED_DATA_DIR, "metadata.yaml")
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f)
    print("Metadata saved to", metadata_path)

    print("Conversion complete.")
    return True

    return True

def main():
    # Check if the processed dataset is already present.
    if check_processed_dataset():
        print("Processed dataset is already present.")
        return

    # Ensure raw AMC file is available.
    if not os.path.exists(RAW_DATA_DIR) or not os.path.exists(os.path.join(RAW_DATA_DIR, "06_14.amc")):
        print("Raw AMC file not found. Downloading...")
        if not download_raw_amc():
            print("Failed to download raw AMC file.")
            return

    # Convert the raw dataset into the processed format.
    if not convert_cmu_to_deepmimic_format():
        print("Conversion failed.")
        return

    if check_processed_dataset():
        print("Dataset is ready for use.")
    else:
        print("Dataset conversion did not produce all required files.")

if __name__ == "__main__":
    main()
