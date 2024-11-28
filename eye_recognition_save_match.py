import cv2
import numpy as np
from scipy.spatial.distance import euclidean
import time
from skimage.feature import hog
from sklearn.preprocessing import normalize
import os

# Feature extraction using HOG (Histogram of Oriented Gradients)
def extract_hog_features(image):
    # Check if the image has 3 channels (BGR). If so, convert to grayscale.
    if len(image.shape) == 3 and image.shape[2] == 3:  # If image is in color
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image  # Image is already grayscale

    # Resize to a fixed size to ensure consistency
    resized_img = cv2.resize(gray_image, (64, 128))
    
    # Extract HOG features
    features, _ = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

# Normalize features to improve matching consistency
def normalize_features(features):
    return normalize([features])[0]

# Match features between input and saved data using Euclidean distance
def match_features(input_features, saved_features):
    distance = euclidean(input_features, saved_features)
    return distance

# Step 1: Capture iris data, save image, and extract features
def capture_and_save_image(filename="saved_iris_image.jpg"):
    cap = cv2.VideoCapture(0)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    print("Position your eye in front of the camera. Capturing and saving iris image...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error. Exiting.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in eyes:
            eye_region = frame[y:y + h, x:x + w]

            # Save the captured eye image
            cv2.imwrite(filename, eye_region)
            print(f"Iris image captured and saved as {filename}")

            # Extract HOG features from the saved eye image
            iris_features = extract_hog_features(eye_region)

            # Normalize the extracted features and save them
            normalized_iris_features = normalize_features(iris_features)
            np.save("saved_iris_features.npy", normalized_iris_features)

            cap.release()
            cv2.destroyAllWindows()
            return  # Exit function after saving data

        # Show live feed
        cv2.imshow("Capture Iris Image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Step 2: Match real-time iris data with saved data
def match_iris_data(filename="saved_iris_features.npy"):
    cap = cv2.VideoCapture(0)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    try:
        saved_features = np.load(filename)
        print("Saved iris features loaded.")
    except FileNotFoundError:
        print("No saved data found. Please capture data first.")
        return

    print("Matching real-time input with saved iris data. This will run for 1 minute. Press 'q' to quit early.")

    start_time = time.time()
    match_count = 0
    unmatch_count = 0
    match_threshold = 0.5  # Distance threshold for matching (lower is stricter)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error. Exiting.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in eyes:
            eye_region = gray[y:y + h, x:x + w]

            # Extract HOG features from the detected eye region
            iris_features = extract_hog_features(eye_region)

            # Normalize the extracted features
            normalized_iris_features = normalize_features(iris_features)

            # Match the features using Euclidean distance
            distance = match_features(normalized_iris_features, saved_features)

            # Determine if the match is valid based on the threshold
            result = "Match" if distance < match_threshold else "No Match"
            if result == "Match":
                match_count += 1
            else:
                unmatch_count += 1

            # Display results on the frame
            cv2.putText(frame, f"Result: {result}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Distance: {round(distance, 2)}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            print(f"[{time.strftime('%H:%M:%S')}] Eye detected. {result} (Distance: {round(distance, 2)})")

        # Show live feed with results
        cv2.imshow("Match Iris Data", frame)

        # Exit after 1 minute or on 'q'
        if time.time() - start_time > 60 or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print summary to terminal
    print("\n=== Matching Summary ===")
    print(f"Matches: {match_count}")
    print(f"Unmatches: {unmatch_count}")

# Main function to orchestrate the process
if __name__ == "__main__":
    print("Automated Eye Recognition System")

    # Step 1: Capturing and saving iris image
    print("Step 1: Capturing and saving iris image...")
    capture_and_save_image()

    # Step 2: Matching real-time input with saved data
    print("\nStep 2: Matching real-time input with saved data...")
    match_iris_data()
