#RUN FIRST TO APPLY KMEANS ON IMAGES
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from PIL import Image

def image_to_1d_array(image_path):
    # Load the image
    print("Loading image...")
    image = Image.open(image_path)
    # Get the width and height of the image
    width, height = image.size
    print("Image size:", width, "x", height)
    # Convert the image to a numpy array
    print("Converting image to numpy array...")
    image_array = np.array(image)
    
    # Extract white pixels and their coordinates
    white_pixels = []
    print("Extracting white pixels...")
    for y in range(height):
        for x in range(width):
            # Check if pixel is white (you can adjust the threshold as needed)
            if np.all(image_array[y, x] >= 240):  # Assuming white is [240, 240, 240]
                # Append pixel coordinates (x, y) and color values separately
                white_pixels.append([y, x, *image_array[y, x]])
    white_pixels_array = np.array(white_pixels)
    print("Final array before K-means clustering:")
    print(white_pixels_array)
    return white_pixels_array

def find_optimal_clusters(data, max_clusters=500):
    silhouette_scores = []
    cluster_heads = []
    print("Finding optimal number of clusters...")
    for n_clusters in range(2, min(max_clusters, len(data) // 2) + 1):
        print("Running KMeans for", n_clusters, "clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data[:, :2])  # Clustering based only on coordinates
        silhouette_avg = silhouette_score(data[:, :2], cluster_labels)
        silhouette_scores.append((n_clusters, silhouette_avg))
        cluster_heads.append(kmeans.cluster_centers_)
        if n_clusters >= 490 and n_clusters % 10 == 0:
            visualize_clusters(data, cluster_labels, image_path, n_clusters)
    
    # Get the optimal number of clusters based on maximum silhouette score
    optimal_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
    
    # Compute additional metrics after fitting k-means with optimal clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    kmeans.fit(data[:, :2])  # Fit k-means using optimal number of clusters
    inertia = kmeans.inertia_
    db_index = davies_bouldin_score(data[:, :2], kmeans.labels_)
    
    # Plot silhouette scores
    plot_silhouette_scores(silhouette_scores)

    return optimal_clusters, silhouette_scores, cluster_heads, inertia, db_index

def visualize_clusters(data, cluster_labels, image_path, n_clusters):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    unique_labels = np.unique(cluster_labels)
    cluster_centers = []
    for label in unique_labels:
        cluster_data = data[cluster_labels == label]
        cluster_center = np.mean(cluster_data, axis=0)
        cluster_centers.append(cluster_center)
    cluster_centers = np.array(cluster_centers)
    plt.scatter(cluster_centers[:, 1], cluster_centers[:, 0], c='blue', s=50, marker='x')
    plt.title(f'Clusters for {n_clusters} clusters')
    plt.show()

def plot_silhouette_scores(silhouette_scores):
    k_values, silhouette_values = zip(*silhouette_scores)
    plt.plot(k_values, silhouette_values, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different Numbers of Clusters')
    plt.show()

#CHANGE PATH ACCORDING TO IMAGE WANTED TO APPLY KMEANS ON
image_path = 'datasets/RFCP/0006179.png'

# Convert image to 1D array containing only white pixel coordinates and color values
white_pixel_data = image_to_1d_array(image_path)

# Find optimal number of clusters, silhouette scores, cluster centroids, inertia, and Davies-Bouldin index
optimal_clusters, silhouette_scores, cluster_heads, inertia, db_index = find_optimal_clusters(white_pixel_data)

# Calculate mean of all silhouette scores
mean_silhouette_score = np.mean([score for _, score in silhouette_scores])

# Print results
print("Optimal number of clusters:", optimal_clusters)
print("Mean silhouette score for all clusters:", mean_silhouette_score)
print("Silhouette scores for each number of clusters:")
for n_clusters, silhouette_score in silhouette_scores:
    print("Number of clusters:", n_clusters, "- Silhouette score:", silhouette_score)

print("\nAdditional metrics:")
print("Inertia:", inertia)
print("Davies-Bouldin Index:", db_index)
print("mean avg silhouette: ",mean_silhouette_score )
