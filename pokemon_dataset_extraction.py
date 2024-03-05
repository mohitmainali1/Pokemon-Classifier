import cv2
import os
import numpy as np
import csv

def extract_features(image):
    # Color histogram calculations done with the highest possible bin size for each channel.
    # Set to max if the user decides to add their own dataset with better images.
    hist = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Gray-scaled image to reduce dimensionality for contour, pixel intensities, and shapes.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contours, ignore00 = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)

    # Pixel intensities.
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    # Calculate spatial features (width, height)
    height, width = image.shape[:2]

    return hist, num_contours, mean_intensity, std_intensity, width, height

def extract_pokemon_name(filename):
    # Helper to write Pokémon names from filename
    pokemon_name = filename.split("-")[0].capitalize()
    pokemon_name_no_extension = os.path.splitext(pokemon_name)[0]
    return pokemon_name_no_extension

def main(images_folder, output_file):
    features = []
    pokemon_names = []

    for filename in os.listdir(images_folder):
        if filename.endswith('.png') or filename.endswith('.PNG'):
            image_path = os.path.join(images_folder, filename)

            # Read the image
            image = cv2.imread(image_path)
            if image is not None:
                hist, num_contours, mean_intensity, std_intensity, width, height = extract_features(image)
                pokemon_name = extract_pokemon_name(filename)

                # Append features and Pokémon name to the lists
                features.append([hist, num_contours, mean_intensity, std_intensity, width, height])
                pokemon_names.append(pokemon_name)

    # Write features that were extracted into our training CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Color Histogram", "Num Contours", "Mean Intensity", "Std Intensity", "Width", "Height", "Pokemon Name"])
        for feature, name in zip(features, pokemon_names):
            writer.writerow([feature[0], feature[1], feature[2], feature[3], feature[4], feature[5], name])

    print(f"Dataset saved to '{output_file}'")

if __name__ == "__main__":
    images_folder = "images"
    output_file = "pokemon_dataset.csv"
    main(images_folder, output_file)