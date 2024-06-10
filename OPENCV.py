import cv2
import matplotlib.pyplot as plt
# Read an image from file
image = cv2.imread('C://Users//Administration//OneDrive//Desktop//Open cv//Jackal.jpg')
# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Display the image in a window
cv2.imshow('Image', image)
# Perform edge detection
edges = cv2.Canny(gray_image, 100, 200)
# Display the original image
cv2.imshow('Original Image', image)
# Display the grayscale image
cv2.imshow('Grayscale Image', gray_image)
# Display the edges detected
cv2.imshow('Edges', edges)
# Perform object detection (face detection)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
# Display the image with detected faces
cv2.imshow('Image with Detected Faces', image)
# Perform feature extraction (ORB feature detector)
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(gray_image, None)
# Draw keypoints on the original image
keypoints_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Image with ORB Keypoints', keypoints_image)
# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Display the original and grayscale images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title("Grayscale Image")
plt.imshow(gray_image, cmap='gray')

plt.show()
