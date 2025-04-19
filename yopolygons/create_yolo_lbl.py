import os
import argparse
from yolo import (
    load_mask_from_file,
    mask_to_yolo_segmentation,
    test_polygon_to_mask_display,
)

if __name__ == "__main__":
    # Define source and destination folders depending on the operating system
    parser = argparse.ArgumentParser(description="Convert PNG masks to YOLO segmentation labels.")
    parser.add_argument("-s", "--source", type=str, help="Path to the folder containing PNG mask files.")
    parser.add_argument("-ep", "--epsilon_ratio", type=float, default=0.005, help="Epsilon ratio for polygon approximation (default: 0.005).")
    parser.add_argument("-v", "--visualize", action="store_true", help="Enable visualization of the resulting labels.")
    args = parser.parse_args()

    # Define source folder
    if args.source:
        src_folder = args.source
    else:
        src_folder = input("Please enter the path to the folder containing PNG mask files: ").strip()

    dest_folder = os.path.join(os.getcwd(), "labels")

    # Create destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Loop through files in the source folder
    for file_name in os.listdir(src_folder):
        if file_name.endswith('.png'):
            file_path = os.path.join(src_folder, file_name)

            # Load the binary mask
            binary_mask = load_mask_from_file(file_path)

            if binary_mask is None:
                continue

            lbl_file = os.path.join(dest_folder, f'{os.path.splitext(file_name)[0]}.txt')
            cls_id = 0  # Replace with actual class ID

            # Convert mask to YOLO segmentation format and save
            mask_to_yolo_segmentation(binary_mask, cls_id, lbl_file, epsilon_ratio=args.epsilon_ratio)

    # Test the polygon to mask display for each generated label file if visualization is enabled
    if args.visualize:
        for file_name in os.listdir(src_folder):
            if file_name.endswith('.png'):
                file_path = os.path.join(src_folder, file_name)

                # Load the binary mask to get dimensions
                binary_mask = load_mask_from_file(file_path)

                if binary_mask is None:
                    continue

                img_height, img_width = binary_mask.shape
                label_file = os.path.join(dest_folder, f'{os.path.splitext(file_name)[0]}.txt')

                # Display the mask from the label file
                test_polygon_to_mask_display(label_file,
                                             image_width=img_width,
                                             image_height=img_height,
                                             display_separately=False)