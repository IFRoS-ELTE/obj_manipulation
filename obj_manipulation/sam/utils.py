from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def collect_points(img: NDArray) -> Tuple[NDArray, NDArray]:
    """Collect positive and negative point labels for segmentation using SAM.
    
    Args:
        img: [H x W x 3] input RGB image to collect points from.
    
    Returns:
        tuple
        - points: [N x 2] array with the image coordinates of each selected point.
        - labels: [N] array of corresponding labels (0 = ignore, 1 = segment) for each selected
            points.
    """
    # Lists to store coordinates of positive and negative points 
    positive, negative = [], []
    img_display = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.namedWindow("Select Points")
    
    # Define mouse callback to store point coordinates and type according to button pressed
    def mouse_callback(event: int, x: int, y: int, flags: int, params: Any) -> None:
        """Mouse event callback function."""
        if event == cv2.EVENT_LBUTTONDOWN:
            positive.append((x, y))
            cv2.circle(img_display, (x, y), 4, (0, 255, 0), -1)  # Green for positive
        elif event == cv2.EVENT_RBUTTONDOWN:
            negative.append((x, y))
            cv2.circle(img_display, (x, y), 4, (0, 0, 255), -1)  # Red for negative
        cv2.imshow("Select Points", img_display)
    cv2.setMouseCallback("Select Points", mouse_callback)

    # Collect points until ESC is pressed
    print("\nCollecting points. Use left button for positive points and right for negative points.")
    print("Do not choose multiple objects. Press ESC when done to exit.")
    while True:
        cv2.imshow("Select Points", img_display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC pressed
            break
    cv2.destroyAllWindows()

    # Combine all point coordinates and create corresponding labels
    points = np.array(positive + negative)
    labels = np.array([1] * len(positive) + [0] * len(negative))
    return points, labels
