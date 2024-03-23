import argparse
import requests
import os

# ArgumentParser-Instanz erstellen
parser = argparse.ArgumentParser(description="Download files from a specified Dataverse dataset.")
parser.add_argument("--continue", dest="continue_download", action="store_true", 
                    help="Continue downloading files if they already exist")
parser.add_argument("--destination", type=str, default="GenImage_download",
                    help="Destination directory for downloaded files")
parser.set_defaults(continue_download=False)
args = parser.parse_args()

BASE_URL = "https://dataverse.harvard.edu"
dataverse_id = "doi:10.7910/DVN/AKDIHF"
continue_download = args.continue_download    
destination_directory = args.destination

print("Preparing download...")
response = requests.get(f"{BASE_URL}/api/datasets/:persistentId/?persistentId={dataverse_id}")

if response.status_code == 200:
    files = response.json()['data']['latestVersion']['files']
    print("Starting download...")
    
    for i, file_entry in enumerate(files):
        file = file_entry['dataFile']
        file_id = file['id']
        file_path = os.path.join(destination_directory, file_entry.get('directoryLabel', '.'), file['filename'])
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        if continue_download and os.path.exists(file_path):
            print(f"{os.path.normpath(file_path)} already found [{i+1}/{len(files)}]")
            continue
        
        download_response = requests.get(f"{BASE_URL}/api/access/datafile/{file_id}")
        if download_response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(download_response.content)
            print(f"{os.path.normpath(file_path)} was successfully downloaded [{i+1}/{len(files)}]")
        else:
            print(f"\n!!! Error when trying to download {os.path.normpath(file_path)} (id:{file_id}) (QUIT!)")
            
    print("\nFINISHED!")
else:
    print("\n!!! Error when trying to connect to dataverse (QUIT!)")
