import cv2
import numpy as np

# Load the image
img = cv2.imread("outputs/images/patinet_5_2CH_contours_processed.jpg")
cv2.imshow("Image with Rectangle", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Define the size of the rectangle
rect_width = 140
rect_height = 36

# Calculate the coordinates of the top left and bottom right corners of the rectangle
rect_x1 = int(img.shape[1]/2 - rect_width/2 + 140)  # add 140 to center the rectangle horizontally
rect_y1 = int(img.shape[0]/2 - rect_height/2 + 36)  # add 36 to center the rectangle vertically
rect_x2 = rect_x1 + rect_width
rect_y2 = rect_y1 + rect_height

# Create a white rectangle
white_rect = np.zeros_like(img)
white_rect[rect_y1:rect_y2, rect_x1:rect_x2, :] = 255

# Add the white rectangle to the original image
img_with_rect = cv2.add(img, white_rect)

# Display the image
cv2.imshow("Image with Rectangle", img_with_rect)
cv2.waitKey(0)
cv2.destroyAllWindows()
