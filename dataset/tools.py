import os
import shutil

class DatasetOrganizer:
    def __init__(self, dataset_path):
        """
        Initialize the DatasetOrganizer class.
        :param dataset_path: Path to the dataset
        """
        self.dataset_path = dataset_path

    def list_files(self, file_type=None):
        """
        List all files in the dataset. If a file type is specified, only list files of that type.
        :param file_type: Specify the file extension (e.g., 'jpg' or 'txt')
        :return: List of file names
        """
        files = []
        for root, dirs, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if file_type:
                    if filename.endswith(file_type):
                        files.append(os.path.join(root, filename))
                else:
                    files.append(os.path.join(root, filename))
        return files

    def rename_files(self, prefix, file_type=None):
        """
        Rename files in the dataset using the specified prefix, keeping the file extension.
        :param prefix: New file name prefix
        :param file_type: The file type to rename (e.g., 'jpg' or 'txt')
        """
        files = self.list_files(file_type)
        for i, file in enumerate(files):
            directory, original_filename = os.path.split(file)
            extension = os.path.splitext(original_filename)[1]
            new_filename = f"{prefix}_{i}{extension}"
            new_filepath = os.path.join(directory, new_filename)
            os.rename(file, new_filepath)

    def categorize_by_extension(self):
        """
        Categorize files into different subfolders based on their file extension.
        """
        files = self.list_files()
        for file in files:
            extension = os.path.splitext(file)[1][1:]  # Get file extension without dot
            ext_folder = os.path.join(self.dataset_path, extension)
            if not os.path.exists(ext_folder):
                os.makedirs(ext_folder)
            shutil.move(file, os.path.join(ext_folder, os.path.basename(file)))

    def remove_empty_files(self):
        """
        Remove empty files from the dataset.
        """
        files = self.list_files()
        for file in files:
            if os.path.getsize(file) == 0:
                os.remove(file)
                print(f"Removed empty file: {file}")

    def remove_files_by_extension(self, file_type):
        """
        Remove files of a specified type.
        :param file_type: The file type to remove (e.g., 'jpg', 'txt')
        """
        files = self.list_files(file_type)
        for file in files:
            os.remove(file)
            print(f"Removed file: {file}")
    
    def copy_files_to_folder(self, destination_folder, file_type=None):
        """
        Copy specified types of files from the dataset to a destination folder.
        :param destination_folder: The destination folder path
        :param file_type: The file type to copy (e.g., 'jpg' or 'txt')
        """
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        
        files = self.list_files(file_type)
        for file in files:
            shutil.copy(file, destination_folder)
            print(f"Copied {file} to {destination_folder}")

# Usage example
if __name__ == "__main__":
    dataset_path = "/path/to/your/dataset"
    organizer = DatasetOrganizer(dataset_path)
    
    # List all JPG files
    print(organizer.list_files("jpg"))
    
    # Rename all JPG files
    organizer.rename_files(prefix="image", file_type="jpg")
    
    # Categorize files by extension
    organizer.categorize_by_extension()
    
    # Remove empty files
    organizer.remove_empty_files()
    
    # Remove all TXT files
    organizer.remove_files_by_extension("txt")
    
    # Copy all PNG files to another folder
    organizer.copy_files_to_folder("/path/to/destination/folder", file_type="png")
