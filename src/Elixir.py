import cv2
import numpy as np

class Elixir:
    """
    Class responsible for extracting and processing elixir
    """

    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.roi = self._setup_roi()
        self._setup_hsv_range()

    def _setup_roi(self):
        """
        Set up the region of interest (ROI) for elixir detection, on Clash Royale it's on the bottom center of the screen.
        Returns a tuple (x_start, y_start, x_end, y_end) defining the ROI coordinates.
        """
        self.roi_x1 = int(self.frame_width * 0.05)
        self.roi_y1 = int(self.frame_height * 0.2)
        self.roi_x2 = int(self.frame_width * 0.85)
        self.roi_y2 = int(self.frame_height * 0.95)

        return (self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2)

    def _setup_hsv_range(self)
        """
        Setup the HSV Color range for elixir detection, HSV is better than RGB for color detection.
        Purple on HSV:
        - Hue: 130-150 (Purple range)
        - Saturation: 100-255 (High saturation)
        - Value: 100-255 (Bright colors)
        """

        # Lower bound for purple color in HSV
        self.hsv_lower = np.array([130, 100, 100])

        # Upper bound for purple color in HSV
        self.hsv_upper = np.array([150, 255, 255])
    
    def _extract_roi(self, frame):
        """
        Extract the region of interest (ROI) from the frame.
        """
        x1, y1, x2, y2 = self.roi
        return frame[y1:y2, x1:x2]

    def _detect_purple_pixels(self, roi):
        """
        Detect purple pixels in the ROI using HSV color space.
        
        Args:
            roi (numpy.ndarray): The region of interest image.

        Returns:
            tuple: (mask, purple_ratio)
                - mask (numpy.ndarray): Binary mask where purple pixels are white.
                - purple_ratio (float): Ratio of purple pixels to total pixels in the ROI.
        """

        # Convert from BGR to HSV color space
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Create mask for purple color
        mask = cv2.inRange(hsv_roi, self.hsv_lower, self.hsv_upper)

        # Calculate the ratio of purple pixels to total pixels
        total_pixels = mask.size
        purple_pixels = np.count_nonzero(mask)
        purple_ratio = purple_pixels / total_pixels if total_pixels > 0 else 0

        return mask, purple_ratio

    def _ratio_to_elxixir(self, purple_ratio):
        """
        Convert the purple pixel ratio to elixir amount.
        
        Args:
            purple_ratio (float): Ratio of purple pixels to total pixels in the ROI.

        Returns:
            int: Estimated elixir amount (0-10).
        """
        # Assuming a linear relationship between purple ratio and elixir amount
        elixir_amount = int(purple_ratio * 10)

        return min(max(elixir_amount, 0), 10)

    def get_elixir(self, frame):
        """
        Get the elixir amount from the given frame.
        
        Args:
            frame (numpy.ndarray): The input image frame.
        Returns:
            int: Estimated elixir amount (0-10).
        """

        roi = self._extract_roi(frame)
        mask, purple_ratio = self._detect_purple_pixels(roi)
        elixir_amount = self._ratio_to_elxixir(purple_ratio)

        return elixir_amount

    def get_elixir_with_debug(self, frame):
        """
        Get the elixir amount from the given frame with debug information.
        
        Args:
            frame (numpy.ndarray): The input image frame.
        Returns:
            tuple: (elixir_amount, debug_frame, roi, mask, purple_ratio, debug_info)
                - elixir_amount (int): Estimated elixir amount (0-10).
                - debug_frame (numpy.ndarray): Frame with debug overlays.
                - roi (numpy.ndarray): Extracted region of interest.
                - mask (numpy.ndarray): Binary mask of detected purple pixels.
                - purple_ratio (float): Ratio of purple pixels to total pixels in the ROI.
                - debug_info (dict): Additional debug information.
                    - roi: The extracted region of interest.
                    - mask: The binary mask of detected purple pixels.
                    - purple_ratio: The ratio of purple pixels to total pixels in the ROI.
        """
        roi = self._extract_roi(frame)
        mask, purple_ratio = self._detect_purple_pixels(roi)
        elixir_amount = self._ratio_to_elxixir(purple_ratio)

        debug_info = {
            "roi": roi,
            "mask": mask,
            "purple_ratio": purple_ratio
        }

        debug_frame = frame.copy()
        cv2.rectangle(debug_frame, (self.roi_x1, self.roi_y1), (self.roi_x2, self.roi_y2), (0, 255, 0), 2)
        text = f"Elixir: {elixir_amount}"
        cv2.putText(debug_frame, text, (self.roi_x1, self.roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return elixir_amount, debug_frame, roi, mask, purple_ratio, debug_info