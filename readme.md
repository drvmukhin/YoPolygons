# YOLO Segmentation Mask Converter

## Overview

This project provides a Python package, `YoPolygons`, for converting segmentation masks generated from a YOLO model (e.g., YOLOv8 or YOLOv11) into YOLO-compatible polygon segmentation labels. The primary goal is to extract contours from binary masks, reduce the number of points using polygon approximation, and normalize the coordinates to create label files suitable for training YOLO segmentation models. The project relies on OpenCV and NumPy for processing.

## Key Features

- **Binary Mask Processing**: Extract contours from segmentation masks using OpenCV.
- **Polygon Approximation**: Simplify contours while preserving the shape using the Ramer-Douglas-Peucker algorithm.
- **Coordinate Normalization**: Convert polygon points to a format compatible with YOLO, normalized between 0 and 1.
- **YOLO Label Generation**: Save the results in YOLO segmentation format (class-id x1 y1 x2 y2 ... xn yn).
- **Visualization Tool**: Display YOLO segmentation masks visually for testing purposes.

## Installation

### Prerequisites

- Python 3.10+
- OpenCV
- NumPy

## Installation

### Install the package:

1. Clone this repository:
   ```sh
   git clone https://github.com/drvmukhin/YoPolygons.git
   cd YoPolygons
   ```

2. Install the package:
   ```sh
   pip install .
   ```

   Alternatively, you can install the dependencies manually:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Running the Conversion Script

This project allows you to convert binary masks into YOLO segmentation label files.

1. Set the paths for your source folder containing the mask images and specify the destination folder for the label files.
2. Run the main script:
   ```sh
   python createe_yolo_lbl.py -v
   ```
   This script will iterate through all the PNG mask files in the source folder, convert them to YOLO format, and save the label files to the destination folder.
   -v parameter executes the `test_polygon_to_mask_display` function provides a way to visualize the segmentation masks generated from the YOLO label files to verify the conversion.
   It will display each mask to visually verify the correctness of the labels.

## Code Structure

- **yolo.py**: Main script for converting binary masks into YOLO-compatible segmentation labels and visualizing them.
- **load\_mask\_from\_file(path\_to\_file)**: Loads a binary mask from a PNG file.
- **mask\_to\_yolo\_segmentation(mask, class\_id, output\_file, epsilon\_ratio)**: Converts a binary mask into YOLO segmentation format and saves it.
- **test\_polygon\_to\_mask\_display(input\_file, image\_width, image\_height, display\_separately)**: Visualizes YOLO segmentation labels as masks.

## Example

1. Place your mask images (e.g., `mask.png`) in the source folder.
2. Run the script to generate YOLO label files with an optional `-ep` (`--epsilon_ratio`) parameter to control polygon approximation:
   ```sh
   python yopolygons/create_yolo_lbl.py -s <source_folder> -ep <epsilon_ratio>
   ```
   Replace `<source_folder>` with the path to your folder containing the mask images and `<epsilon_ratio>` with a value (default: 0.005) to adjust the level of contour simplification.
3. The label files will be saved in a `labels` subfolder in the current working directory.
4. Visualize the output labels to validate the conversion process.

## License

This project is licensed under the MIT License.

## Contact

For any questions or suggestions, please contact [[vmukhin.dev@gmail.com](mailto\:vmukhin.dev@gmail.com)].




