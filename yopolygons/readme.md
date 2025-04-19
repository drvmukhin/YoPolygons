# YOLO Segmentation Mask Converter

## Overview

This project provides a tool for converting segmentation masks generated from a YOLO model (e.g., YOLOv8 or YOLOv11) into YOLO-compatible polygon segmentation labels. The primary goal is to extract contours from binary masks, reduce the number of points using polygon approximation, and normalize the coordinates to create label files suitable for training YOLO segmentation models. The project relies on OpenCV and NumPy for processing.

## Key Features

- **Binary Mask Processing**: Extract contours from segmentation masks using OpenCV.
- **Polygon Approximation**: Simplify contours while preserving the shape using the Ramer-Douglas-Peucker algorithm.
- **Coordinate Normalization**: Convert polygon points to a format compatible with YOLO, normalized between 0 and 1.
- **YOLO Label Generation**: Save the results in YOLO segmentation format (class-id x1 y1 x2 y2 ... xn yn).
- **Visualization Tool**: Display YOLO segmentation masks visually for testing purposes.

## Installation

### Prerequisites

- Python 3.9+
- OpenCV
- NumPy

### Setup

1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/yolo-segmentation-converter.git
   cd yolo-segmentation-converter
   ```
2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Running the Conversion Script

This project allows you to convert binary masks into YOLO segmentation label files.

1. Set the paths for your source folder containing the mask images and specify the destination folder for the label files.
2. Run the main script:
   ```sh
   python yolo_segmentation_converter.py
   ```
   This script will iterate through all the PNG mask files in the source folder, convert them to YOLO format, and save the label files to the destination folder.

### Visualizing Segmentation Masks

The `test_polygon_to_mask_display` function provides a way to visualize the segmentation masks generated from the YOLO label files to verify the conversion.

1. Adjust the paths and run the visualization part in the script:
   ```sh
   python yolo_segmentation_converter.py
   ```
   It will display each mask to visually verify the correctness of the labels.

## Code Structure

- **yolo\_segmentation\_converter.py**: Main script for converting binary masks into YOLO-compatible segmentation labels.
- **extract\_contours\_from\_mask(mask)**: Extracts contours from a given binary mask.
- **approximate\_polygon(contour, epsilon\_ratio)**: Simplifies the extracted contour using the Ramer-Douglas-Peucker algorithm.
- **normalize\_points(points, image\_width, image\_height)**: Normalizes points based on image dimensions.
- **save\_yolo\_segmentation(output\_file, class\_id, normalized\_points)**: Saves the segmentation in YOLO format.
- **test\_polygon\_to\_mask\_display(input\_file, image\_width, image\_height)**: Visualizes the YOLO segmentation labels.

## Example

1. Place your mask images in the source folder (e.g., `salvage_lab`).
2. Run the script to generate YOLO label files:
   ```sh
   python yolo_segmentation_converter.py
   ```
3. The label files will be saved in a `labels` subfolder in the current working directory.
4. Visualize the output labels to validate the conversion process.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please contact [[vmukhin.dev@example.com](mailto\:vmukhin.dev@gmail.com)].




