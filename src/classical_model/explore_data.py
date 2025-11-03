"""
Quick script to explore BraTS dataset structure
"""
import os
import nibabel as nib
import numpy as np

# Paths
images_dir = '../../data/raw/imagesTr'
labels_dir = '../../data/raw/labelsTr'

# Get file lists (filter out Mac OS metadata files)
image_files = sorted([f for f in os.listdir(images_dir) if not f.startswith('._') and f.endswith('.nii.gz')])
label_files = sorted([f for f in os.listdir(labels_dir) if not f.startswith('._') and f.endswith('.nii.gz')])

print(f"Number of images: {len(image_files)}")
print(f"Number of labels: {len(label_files)}")
print(f"\nFirst few image files:")
for f in image_files[:5]:
    print(f"  {f}")

# Load a sample image and label to understand structure
if image_files:
    sample_img_path = os.path.join(images_dir, image_files[0])
    sample_lbl_path = os.path.join(labels_dir, label_files[0])

    img = nib.load(sample_img_path)
    lbl = nib.load(sample_lbl_path)

    img_array = img.get_fdata()
    lbl_array = lbl.get_fdata()

    print(f"\nSample Image Info:")
    print(f"  File: {image_files[0]}")
    print(f"  Shape: {img_array.shape}")
    print(f"  Data type: {img_array.dtype}")
    print(f"  Value range: [{img_array.min():.2f}, {img_array.max():.2f}]")
    print(f"  Voxel dimensions: {img.header.get_zooms()}")

    print(f"\nSample Label Info:")
    print(f"  File: {label_files[0]}")
    print(f"  Shape: {lbl_array.shape}")
    print(f"  Data type: {lbl_array.dtype}")
    print(f"  Unique values: {np.unique(lbl_array)}")
    print(f"  Label distribution:")
    for val in np.unique(lbl_array):
        count = np.sum(lbl_array == val)
        percent = 100 * count / lbl_array.size
        print(f"    Class {val}: {count} pixels ({percent:.2f}%)")
