import os
import shutil
import random
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def resplit_dataset(source_dir, target_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Resplit the dataset into train, validation, and test sets with specified ratios.
    
    Args:
        source_dir (str): Path to the original dataset directory
        target_dir (str): Path to save the resplit dataset
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
        seed (int): Random seed for reproducibility
    """
    # Set random seed
    random.seed(seed)
    
    # Create target directory structure
    for split in ['train', 'val', 'test']:
        for class_name in ['NORMAL', 'PNEUMONIA']:
            os.makedirs(os.path.join(target_dir, split, class_name), exist_ok=True)
    
    # Get all images and their labels
    all_images = []
    for class_name in ['NORMAL', 'PNEUMONIA']:
        class_dir = os.path.join(source_dir, 'train', class_name)
        for img in os.listdir(class_dir):
            all_images.append((os.path.join(class_dir, img), class_name))
    
    # Shuffle the dataset
    random.shuffle(all_images)
    
    # Calculate split indices
    n_images = len(all_images)
    train_end = int(n_images * train_ratio)
    val_end = train_end + int(n_images * val_ratio)
    
    # Split the dataset
    train_images = all_images[:train_end]
    val_images = all_images[train_end:val_end]
    test_images = all_images[val_end:]
    
    # Copy files to new locations
    for split, images in [('train', train_images), ('val', val_images), ('test', test_images)]:
        for img_path, class_name in images:
            filename = os.path.basename(img_path)
            target_path = os.path.join(target_dir, split, class_name, filename)
            shutil.copy2(img_path, target_path)
    
    return train_images, val_images, test_images

def visualize_split_distribution(train_images, val_images, test_images):
    """
    Visualize the distribution of classes in each split.
    
    Args:
        train_images (list): List of (image_path, class_name) tuples for training set
        val_images (list): List of (image_path, class_name) tuples for validation set
        test_images (list): List of (image_path, class_name) tuples for test set
    """
    # Count classes in each split
    train_counts = Counter([img[1] for img in train_images])
    val_counts = Counter([img[1] for img in val_images])
    test_counts = Counter([img[1] for img in test_images])
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot training set distribution
    sns.barplot(x=list(train_counts.keys()), y=list(train_counts.values()), ax=ax1)
    ax1.set_title('Training Set Distribution')
    ax1.set_ylabel('Number of Images')
    
    # Plot validation set distribution
    sns.barplot(x=list(val_counts.keys()), y=list(val_counts.values()), ax=ax2)
    ax2.set_title('Validation Set Distribution')
    ax2.set_ylabel('Number of Images')
    
    # Plot test set distribution
    sns.barplot(x=list(test_counts.keys()), y=list(test_counts.values()), ax=ax3)
    ax3.set_title('Test Set Distribution')
    ax3.set_ylabel('Number of Images')
    
    plt.tight_layout()
    plt.savefig('dataset_distribution.png')
    plt.close()

if __name__ == "__main__":
    # Define paths
    source_dir = r"C:\Users\abdal\.cursor\Projects\Xhassan\mlops-xray-classifier\chest_xray"
    target_dir = r"C:\Users\abdal\.cursor\Projects\Xhassan\mlops-xray-classifier\splited"
    
    # Resplit the dataset
    train_images, val_images, test_images = resplit_dataset(source_dir, target_dir)
    
    # Visualize the distribution
    visualize_split_distribution(train_images, val_images, test_images)
    
    # Print statistics
    print("\nDataset Split Statistics:")
    print(f"Total images: {len(train_images) + len(val_images) + len(test_images)}")
    print(f"Training set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")
    print(f"Test set: {len(test_images)} images") 