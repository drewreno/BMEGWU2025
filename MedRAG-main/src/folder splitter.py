import os
import shutil

def flatten_directory(src_dir, dest_dir):
    """
    Recursively copy all files from src_dir (including subfolders)
    into dest_dir. If duplicate filenames exist, a counter is appended
    to the file name to avoid overwriting.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created destination folder: {dest_dir}")
    
    for root, _, files in os.walk(src_dir):
        for file in files:
            src_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(dest_dir, file)
            
            # If a file with the same name exists in the destination,
            # modify the filename by appending a counter.
            if os.path.exists(dest_file_path):
                base, ext = os.path.splitext(file)
                counter = 1
                new_file = f"{base}_{counter}{ext}"
                dest_file_path = os.path.join(dest_dir, new_file)
                while os.path.exists(dest_file_path):
                    counter += 1
                    new_file = f"{base}_{counter}{ext}"
                    dest_file_path = os.path.join(dest_dir, new_file)
            
            shutil.copy2(src_file_path, dest_file_path)
            print(f"Copied: {src_file_path} -> {dest_file_path}")

if __name__ == "__main__":
    # Ask the user for the source directory and the destination directory.
    src_directory = input("Enter the source directory: ").strip()
    dest_directory = input("Enter the destination directory: ").strip()

    # Validate the source directory
    if not os.path.isdir(src_directory):
        print("Error: The source directory does not exist or is not a directory.")
    else:
        flatten_directory(src_directory, dest_directory)
        print("All files have been copied to the destination folder.")
