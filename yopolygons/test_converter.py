import unittest
import numpy as np
import os
import cv2
from yopolygons.yolo import (
    extract_contours_from_mask,
    approximate_polygon,
    normalize_points,
    save_yolo_segmentation,
    load_mask_from_file,
    mask_to_yolo_segmentation,
)

class TestYoPolygons(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = "test_data"
        os.makedirs(self.test_dir, exist_ok=True)

        # Create a simple binary mask for testing
        self.test_mask = np.zeros((100, 100), dtype=np.uint8)
        self.test_mask[25:75, 25:75] = 255  # A square in the center

        # Save the mask to a file
        self.test_mask_file = os.path.join(self.test_dir, "test_mask.png")
        cv2.imwrite(self.test_mask_file, self.test_mask)

        # Define a test output file
        self.test_output_file = os.path.join(self.test_dir, "test_output.txt")

    def tearDown(self):
        # Clean up test files
        if os.path.exists(self.test_dir):
            for file in os.listdir(self.test_dir):
                os.remove(os.path.join(self.test_dir, file))
            os.rmdir(self.test_dir)

    def test_extract_contours_from_mask(self):
        contours = extract_contours_from_mask(self.test_mask)
        self.assertEqual(len(contours), 1)  # Expect one contour
        self.assertGreater(len(contours[0]), 0)  # Contour should have points

    def test_approximate_polygon(self):
        contours = extract_contours_from_mask(self.test_mask)
        approx = approximate_polygon(contours[0], epsilon_ratio=0.01)
        self.assertGreater(len(approx), 0)  # Approximation should have points

    def test_normalize_points(self):
        contours = extract_contours_from_mask(self.test_mask)
        approx = approximate_polygon(contours[0], epsilon_ratio=0.01)
        normalized = normalize_points(approx, 100, 100)
        self.assertTrue(all(0 <= p <= 1 for p in normalized))  # Points should be normalized

    def test_save_yolo_segmentation(self):
        contours = extract_contours_from_mask(self.test_mask)
        approx = approximate_polygon(contours[0], epsilon_ratio=0.01)
        normalized = normalize_points(approx, 100, 100)
        save_yolo_segmentation(self.test_output_file, 0, normalized)
        self.assertTrue(os.path.exists(self.test_output_file))  # File should be created

    def test_load_mask_from_file(self):
        mask = load_mask_from_file(self.test_mask_file)
        self.assertIsNotNone(mask)  # Mask should be loaded
        self.assertEqual(mask.shape, (100, 100))  # Shape should match

    def test_mask_to_yolo_segmentation(self):
        mask_to_yolo_segmentation(self.test_mask, 0, self.test_output_file, epsilon_ratio=0.01)
        self.assertTrue(os.path.exists(self.test_output_file))  # File should be created

if __name__ == "__main__":
    unittest.main()