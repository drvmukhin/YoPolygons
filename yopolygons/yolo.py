import cv2
import numpy as np

def extract_contours_from_mask(mask, min_area=10):
    """
    Extract contours from a binary mask using OpenCV.
    :param min_area: Minimum area of contours to be considered valid.
    :param mask: Binary mask as a NumPy array.
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

def load_img_from_file(path_to_file):
    """
    Load an image from a file as an RGB image.
    :param path_to_file: Path to the image file.
    :return: RGB image as a NumPy array.
    """
    try:
        img = cv2.imread(path_to_file, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not load file: {path_to_file}")
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def mask_to_yolo_segmentation(mask, class_id, output_file, epsilon_ratio=0.001):
    """
    Convert a binary mask to YOLO segmentation format and save it to a file.
    :param mask: Binary mask as a NumPy array.
    :param class_id: Class ID of the object.
    :param output_file: Output .txt file to save the segmentation.
    :param epsilon_ratio: Approximation parameter for contour reduction.
    :return: Number of contours processed.
    """
    # Get the dimensions of the image
    image_height, image_width = mask.shape[0:2]

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

    return len(contours)

def test_polygon_to_mask_display(input_file, image_width, image_height, display_separately=True, img=None):
    """
    Test function to read polygon from a file, convert it into a segmentation mask, and display it visually.
    :param input_file: Path to the YOLO format file containing polygon points.
    :param image_width: Width of the image.
    :param image_height: Height of the image.
    :param display_separately: If True, display each polygon separately.
    :param img: Optional image (numpy array) to overlay the mask on.
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

            # Display the mask, optionally overlaid on the image
            if img is not None:
                # Create a colored overlay for the mask
                overlay = img.copy()
                colored_mask = np.zeros_like(overlay)
                if len(overlay.shape) == 3:  # Color image
                    print(f"Overlay color image shape: {overlay.shape}")
                    # Apply a semi-transparent colored overlay
                    colored_mask[mask == 255] = [0, 0, 255]  # Red color for the mask
                    overlay = cv2.addWeighted(colored_mask, 0.5, overlay, 1, 0, overlay)
                    cv2.imshow("Segmentation Overlay", overlay)
                else:  # Grayscale image
                    print("Overlay grayscale image")
                    # Convert grayscale image to BGR for overlay
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    # Create colored mask
                    colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    colored_mask[mask == 255] = [0, 0, 255]  # Red color for the mask
                    # Combine image and mask with transparency
                    result = cv2.addWeighted(img_bgr, 1, colored_mask, 0.5, 0)
                    cv2.imshow("Segmentation Overlay", result)
            else:
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

        # Display the merged mask, optionally overlaid on the image
        if img is not None:
            # Create a colored overlay for the mask
            overlay = img.copy()
            colored_mask = np.zeros_like(overlay)
            if len(overlay.shape) == 3:  # Color image
                print(f"Overlay color image shape: {overlay.shape}")
                # Apply a semi-transparent colored overlay
                colored_mask[mask == 255] = [0, 0, 255]  # Red color for the mask
                overlay = cv2.addWeighted(colored_mask, 0.5, overlay, 1, 0, overlay)
                cv2.imshow("Segmentation Overlay", overlay)
            else:  # Grayscale image
                print("Overlay grayscale image")
                # Convert grayscale image to BGR for overlay
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                # Create colored mask
                colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                colored_mask[mask == 255] = [0, 0, 255]  # Red color for the mask
                # Combine image and mask with transparency
                result = cv2.addWeighted(img_bgr, 1, colored_mask, 0.5, 0)
                cv2.imshow("Segmentation Overlay", result)
        else:
            cv2.imshow("Segmentation Mask", mask)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
