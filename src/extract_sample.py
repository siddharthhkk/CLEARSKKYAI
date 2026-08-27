import tarfile
import os

data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

def extract_and_cleanup(archive_name):
    archive_path = os.path.join(data_dir, archive_name)
    if os.path.exists(archive_path):
        print(f"Extracting {archive_name}...")
        
        # Handle tar extraction safely across Python versions
        with tarfile.open(archive_path, "r:gz") as tar:
            if hasattr(tarfile, 'data_filter'):
                tar.extractall(path=data_dir, filter='data')
            else:
                tar.extractall(path=data_dir)
                
        print(f"Finished extracting {archive_name}!")
        
        # Delete compressed archive immediately to free up disk space
        print(f"Deleting archive {archive_name} to reclaim disk space...")
        os.remove(archive_path)
        print(f"Reclaimed disk space! Removed {archive_name}.")
    else:
        print(f"Archive {archive_name} not found (already extracted or deleted).")

if __name__ == "__main__":
    # Process S1 (if still present)
    extract_and_cleanup("s1_africa.tar.gz")
    
    # Process S2
    extract_and_cleanup("s2_africa.tar.gz")