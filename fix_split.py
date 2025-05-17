import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

def create_directory_structure(base_path):
    """Create train, validation, and test directories with class subdirectories."""
    splits = ['train', 'val', 'test']
    classes = ['NORMAL', 'PNEUMONIA']
    
    for split in splits:
        for class_name in classes:
            os.makedirs(os.path.join(base_path, split, class_name), exist_ok=True)

def split_dataset(source_dir, target_dir, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split the chest X-ray dataset into train, validation, and test sets.
    
    Parameters:
    -----------
    source_dir : str
        Path to the source directory containing the dataset
    target_dir : str
        Path where the split dataset will be saved
    test_size : float, default=0.15
        Proportion of the dataset to include in the test split
    val_size : float, default=0.15
        Proportion of the dataset to include in the validation split
    random_state : int, default=42
        Random state for reproducibility
    """
    # Create directory structure
    create_directory_structure(target_dir)
    
    # Process each class
    for class_name in ['NORMAL', 'PNEUMONIA']:
        # Get all image files for the current class
        class_dir = os.path.join(source_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
        
        # First split: separate out the test set
        temp_files, test_files = train_test_split(
            image_files,
            test_size=test_size,
            random_state=random_state
        )
        
        # Second split: split the remaining data into train and validation
        val_size_adjusted = val_size / (1 - test_size)
        train_files, val_files = train_test_split(
            temp_files,
            test_size=val_size_adjusted,
            random_state=random_state
        )
        
        # Copy files to their respective directories
        for files, split_name in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
            for file_name in files:
                src_path = os.path.join(source_dir, class_name, file_name)
                dst_path = os.path.join(target_dir, split_name, class_name, file_name)
                shutil.copy2(src_path, dst_path)
        
        # Print statistics
        print(f"\nClass: {class_name}")
        print(f"Train set: {len(train_files)} images")
        print(f"Validation set: {len(val_files)} images")
        print(f"Test set: {len(test_files)} images")

if __name__ == "__main__":
    # Define paths
    source_dir = "chest_xray/train"  # Original dataset location
    target_dir = "chest_xray_split"  # New directory for split dataset
    
    # Create the split
    split_dataset(source_dir, target_dir)
    
    print("\nDataset splitting completed successfully!")
