import cv2
import numpy as np
import os


def extract_contours_from_mask(mask, min_area=10):
    """
    Extract contours from a binary mask using OpenCV.
    :param min_area: Minimum area of contours to be considered valid.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
    return filtered_contours


def approximate_polygon(contour, epsilon_ratio=0.01):
    """
    Approximate the contour to reduce the number of points.
    :param contour: Contour points from the mask.
    :param epsilon_ratio: Parameter for approximation. Lower values keep more points.
    """
    # Calculate the epsilon for approximation based on the contour perimeter
    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx


def normalize_points(points, image_width, image_height):
    """
    Normalize polygon points based on image dimensions.
    """
    normalized_points = []
    for point in points:
        x, y = point[0]
        x_norm = max(0, min(x / image_width, 1))
        y_norm = max(0, min(y / image_height, 1))
        normalized_points.extend([x_norm, y_norm])
    return normalized_points


def save_yolo_segmentation(output_file, class_id, normalized_points):
    """
    Save YOLO segmentation format to a .txt file.
    """
    with open(output_file, 'w') as f:
        f.write(f"{class_id} " + " ".join(map(str, normalized_points)) + "\n")


def load_mask_from_file(path_to_file):
    """
    Load a mask from a PNG file.
    :param path_to_file: Path to the PNG mask file.
    :return: Binary mask as a NumPy array.
    """
    try:
        mask = cv2.imread(path_to_file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not load file: {path_to_file}")
        return mask
    except Exception as e:
        print(f"Error loading mask: {e}")
        return None


def mask_to_yolo_segmentation(mask, class_id, output_file, epsilon_ratio=0.001):
    """
    Convert a binary mask to YOLO segmentation format and save it to a file.
    :param mask: Binary mask as a NumPy array.
    :param class_id: Class ID of the object.
    :param output_file: Output .txt file to save the segmentation.
    :param epsilon_ratio: Approximation parameter for contour reduction.
    """
    # Get the dimensions of the image
    image_height, image_width = mask.shape

    # Step 1: Extract contours
    contours = extract_contours_from_mask(mask)

    # Step 2: Iterate through each contour
    with open(output_file, 'w') as f:
        for contour in contours:
            # Approximate the contour
            approx_contour = approximate_polygon(contour, epsilon_ratio=epsilon_ratio)

            # Step 3: Normalize the points
            normalized_points = normalize_points(approx_contour, image_width, image_height)

            # Step 4: Save to YOLO format
            f.write(f"{class_id} " + " ".join(map(str, normalized_points)) + "\n")



def test_polygon_to_mask_display(input_file, image_width, image_height, display_separately=True):
    """
    Test function to read polygon from a file, convert it into a segmentation mask, and display it visually.
    :param input_file: Path to the YOLO format file containing polygon points.
    :param image_width: Width of the image.
    :param image_height: Height of the image.
    """
    # Load YOLO segmentation points
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File not found: {input_file}")
        return

    if display_separately:
        for line in lines:
            data = line.strip().split()
            class_id = int(data[0])
            points = list(map(float, data[1:]))

            # Convert normalized points to image coordinates
            polygon_points = []
            for i in range(0, len(points), 2):
                x = int(points[i] * image_width)
                y = int(points[i + 1] * image_height)
                polygon_points.append([x, y])

            # Create an empty mask and draw the polygon
            mask = np.zeros((image_height, image_width), dtype=np.uint8)
            polygon_points_np = np.array([polygon_points], dtype=np.int32)
            cv2.fillPoly(mask, polygon_points_np, 255)

            # Display the mask
            cv2.imshow("Segmentation Mask", mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        # Create an empty mask to merge all polygons
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        for line in lines:
            data = line.strip().split()
            class_id = int(data[0])
            points = list(map(float, data[1:]))

            # Convert normalized points to image coordinates
            polygon_points = []
            for i in range(0, len(points), 2):
                x = int(points[i] * image_width)
                y = int(points[i + 1] * image_height)
                polygon_points.append([x, y])

            # Draw the polygon on the mask
            polygon_points_np = np.array([polygon_points], dtype=np.int32)
            cv2.fillPoly(mask, polygon_points_np, 255)

        # Display the merged mask
        cv2.imshow("Merged Segmentation Mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage

    # Define source and destination folders depending on the operating system
    if os.name == 'nt':
        src_folder = "C:\\Users\\drvmu\\PycharmProjects\\AI\\salvage_lab"
    else:
        src_folder = "/volume/vasily/autodistill/predict5"
    dest_folder = os.path.join(os.getcwd(), "labels")

    # Create destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Loop through files in the source folder
    for file_name in os.listdir(src_folder):
        if file_name.endswith('.png'):
            path_to_file = os.path.join(src_folder, file_name)

            # Load the binary mask
            binary_mask = load_mask_from_file(path_to_file)

            if binary_mask is None:
                continue

            # Get the dimensions of the image

            output_file = os.path.join(dest_folder, f'{os.path.splitext(file_name)[0]}.txt')
            class_id = 0  # Replace with actual class ID

            # Convert mask to YOLO segmentation format and save
            mask_to_yolo_segmentation(binary_mask, class_id, output_file, epsilon_ratio=0.001)

    # Test the polygon to mask display
    # test_file = os.path.join(dest_folder, 'mask_front_54d6d82fb25c4e9ab505f8045a6f39f9_ful.txt')  # Replace with an actual file name
    # test_polygon_to_mask_display(test_file, image_width=640, image_height=480)
    # Test the polygon to mask display for each generated label file
    for file_name in os.listdir(src_folder):
        if file_name.endswith('.png'):
            path_to_file = os.path.join(src_folder, file_name)

            # Load the binary mask to get dimensions
            binary_mask = load_mask_from_file(path_to_file)

            if binary_mask is None:
                continue

            image_height, image_width = binary_mask.shape
            label_file = os.path.join(dest_folder, f'{os.path.splitext(file_name)[0]}.txt')

            # Display the mask from the label file
            test_polygon_to_mask_display(label_file,
                                         image_width=image_width,
                                         image_height=image_height,
                                         display_separately=False)

