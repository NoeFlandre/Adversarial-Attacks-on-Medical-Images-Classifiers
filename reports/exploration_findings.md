# Breast Histopathology Image Findings

## Dataset Overview
- Total patients: 279
- Total images: 277,524
  - Class 0 (non-IDC): 198,738 (71.61%)
  - Class 1 (IDC): 78,786 (28.39%)

## Patient-Level Statistics
- Average images per patient: 994.7
  - Class 0: 712.3 images/patient
  - Class 1: 282.4 images/patient
- Significant class imbalance with non-IDC samples representing over 70% of the dataset

## File Format

Class 0 image info - Size: (50, 50), Mode: RGB, Format: PNG
Class 1 image info - Size: (50, 50), Mode: RGB, Format: PNG

Sample filename: 9036_idx5_x1051_y2401_class0.png
Filename format: patientID_idx_x-coordinate_y-coordinate_class.png