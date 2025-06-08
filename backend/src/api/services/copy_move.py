import cv2
import numpy as np
from PIL import Image
import io
from skimage.feature import ORB, match_descriptors
from skimage.color import rgb2gray
from skimage.measure import ransac
from skimage.transform import AffineTransform
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from sklearn.cluster import DBSCAN
import time
from src.api.utils.logger import logger

def resize_if_needed(image, max_size=1200):
    """Resize image if any dimension is larger than max_size while preserving aspect ratio"""
    width, height = image.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        return image.resize((new_width, new_height), Image.LANCZOS)
    return image

def detect_copy_move_orb(image, min_matches=10, min_cluster_size=3, eps=40):
    """
    Detect copy-move forgery using ORB keypoints and DBSCAN clustering
    
    Args:
        image (PIL.Image): Input image
        min_matches (int): Minimum number of matches to consider forgery
        min_cluster_size (int): Minimum number of points in a cluster
        eps (int): Maximum distance between points in DBSCAN clustering
    
    Returns:
        Tuple[PIL.Image, float, list]: Visualization image, confidence score, and detected regions
    """
    start_time = time.time()
    logger.info("Starting copy-move forgery detection with ORB")
    
    # Resize image if needed
    image = resize_if_needed(image)
    
    # Convert PIL image to OpenCV format
    img_np = np.array(image)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Initialize ORB detector with more features for better detection
    orb = cv2.ORB_create(nfeatures=8000)
    
    # Find keypoints and descriptors
    kp, des = orb.detectAndCompute(img_gray, None)
    
    if des is None or len(kp) < min_matches:
        logger.info(f"Not enough keypoints found ({len(kp) if kp else 0})")
        return image.copy(), 0.0, []
    
    # Use BFMatcher to find all matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des, des)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Filter out self-matches and collect valid matches
    valid_matches = []
    for m in matches:
        # Skip self-matches (same keypoint)
        if m.queryIdx != m.trainIdx:
            # Check if the distance is reasonable
            if m.distance < 55:  # Slightly relaxed threshold
                valid_matches.append(m)
    
    if len(valid_matches) < min_matches:
        logger.info(f"Not enough valid matches found ({len(valid_matches)})")
        return image.copy(), 0.0, []
    
    # Extract matched keypoint pairs
    matched_pairs = []
    for m in valid_matches:
        p1 = kp[m.queryIdx].pt
        p2 = kp[m.trainIdx].pt
        # Calculate distance between points
        dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        # Only consider pairs that are some distance apart (not too close)
        if dist > 20:  # Minimum distance between copied regions
            matched_pairs.append((p1, p2))
    
    if len(matched_pairs) < min_matches:
        logger.info(f"Not enough distant matches found ({len(matched_pairs)})")
        return image.copy(), 0.1, []
    
    # Extract all points for clustering
    all_points = []
    for p1, p2 in matched_pairs:
        all_points.append(p1)
        all_points.append(p2)
    
    # Convert to numpy array
    points = np.array(all_points)
    
    # Apply DBSCAN clustering with improved parameters
    clustering = DBSCAN(eps=eps, min_samples=min_cluster_size).fit(points)
    labels = clustering.labels_
    
    # Number of clusters (excluding noise with label -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    if n_clusters < 2:
        logger.info(f"Not enough clusters found ({n_clusters})")
        return image.copy(), 0.2, []
    
    # Create visualization image
    vis_img = img_np.copy()
    
    # Collect points by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label != -1:  # Ignore noise
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(points[i])
    
    # Filter clusters - must have at least min_cluster_size points
    valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_cluster_size}
    
    if len(valid_clusters) < 2:
        logger.info(f"Not enough valid clusters found ({len(valid_clusters)})")
        return image.copy(), 0.3, []
    
    # Draw the clusters with different colors
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)]
    
    detected_regions = []
    
    # Draw each cluster
    for i, (label, cluster_points) in enumerate(valid_clusters.items()):
        color = colors[i % len(colors)]
        
        # Convert points to int32
        pts = np.array(cluster_points, dtype=np.int32)
        
        # Get min bounding rectangle
        x, y, w, h = cv2.boundingRect(pts)
        
        # Draw rectangle
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 3)
        
        # Store detected region
        detected_regions.append({
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h)
        })
    
    # Calculate confidence based on number of clusters and matches
    # Improved confidence calculation for more accurate results
    n_valid_clusters = len(valid_clusters)
    n_matched_pairs = len(matched_pairs)
    
    # Base confidence on multiple factors
    cluster_factor = min(1.0, (n_valid_clusters - 1) * 0.2)  # More clusters = higher confidence
    pairs_factor = min(1.0, n_matched_pairs / 200)  # More matched pairs = higher confidence
    density_factor = 0.0
    
    # Calculate density factor (how dense are the clusters)
    for cluster_points in valid_clusters.values():
        cluster_size = len(cluster_points)
        pts = np.array(cluster_points, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        area = w * h
        if area > 0:
            density = cluster_size / area
            density_factor = max(density_factor, min(1.0, density * 5000))
    
    # Combine factors with weights
    confidence = min(0.95, 0.3 + (cluster_factor * 0.4) + (pairs_factor * 0.3) + (density_factor * 0.3))
    
    elapsed_time = time.time() - start_time
    logger.info(f"Copy-move detection completed in {elapsed_time:.2f} seconds with confidence {confidence:.2f}")
    
    return Image.fromarray(vis_img), confidence, detected_regions

def detect_copy_move_dct(image, block_size=8, threshold=0.95, eps=5):
    """
    Detect copy-move forgery using DCT coefficients of blocks
    
    Args:
        image (PIL.Image): Input image
        block_size (int): Size of the image blocks
        threshold (float): Similarity threshold
        eps (int): Maximum distance between points in DBSCAN clustering
    
    Returns:
        Tuple[PIL.Image, float, list]: Visualization image, confidence score, and detected regions
    """
    start_time = time.time()
    logger.info("Starting copy-move forgery detection with DCT")
    
    # Resize image if needed
    image = resize_if_needed(image)
    
    # Convert to grayscale
    gray_img = image.convert('L')
    img_array = np.array(gray_img)
    
    h, w = img_array.shape
    
    # Calculate how many blocks we can fit
    h_blocks = h // block_size
    w_blocks = w // block_size
    
    # Initialize list to store DCT features of each block
    features = []
    positions = []
    
    # Extract DCT features for each block
    for i in range(h_blocks):
        for j in range(w_blocks):
            # Extract block
            block = img_array[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            
            # Apply DCT
            dct_block = cv2.dct(np.float32(block))
            
            # Extract a subset of the DCT coefficients as the feature
            # Taking the top-left 4x4 coefficients (lower frequencies)
            feature = dct_block[:4, :4].flatten()
            
            # Store feature and position
            features.append(feature)
            positions.append((j*block_size, i*block_size))  # (x, y)
    
    if len(features) < 2:
        logger.info("Not enough blocks for analysis")
        return image.copy(), 0.0, []
    
    # Convert to numpy array
    features = np.array(features)
    
    # Normalize features
    features = features / (np.max(features) + 1e-10)
    
    # Calculate pairwise distances
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(features, 'euclidean'))
    
    # Find similar blocks
    similar_pairs = []
    for i in range(len(distances)):
        for j in range(i+1, len(distances)):
            if distances[i, j] < (1 - threshold):
                # Check if blocks are far enough apart
                pos_i = positions[i]
                pos_j = positions[j]
                block_distance = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                if block_distance > block_size * 2:  # Minimum distance between copied regions
                    similar_pairs.append((i, j))
    
    if len(similar_pairs) < 3:
        logger.info(f"Not enough similar blocks found ({len(similar_pairs)})")
        return image.copy(), 0.1, []
    
    # Extract all points for clustering
    all_points = []
    for i, j in similar_pairs:
        all_points.append(positions[i])
        all_points.append(positions[j])
    
    # Convert to numpy array
    points = np.array(all_points)
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps*block_size, min_samples=3).fit(points)
    labels = clustering.labels_
    
    # Number of clusters (excluding noise with label -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    if n_clusters < 2:
        logger.info(f"Not enough clusters found ({n_clusters})")
        return image.copy(), 0.2, []
    
    # Create visualization image
    vis_img = np.array(image)
    
    # Collect points by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label != -1:  # Ignore noise
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(points[i])
    
    # Filter clusters - must have at least 3 points
    valid_clusters = {k: v for k, v in clusters.items() if len(v) >= 3}
    
    if len(valid_clusters) < 2:
        logger.info(f"Not enough valid clusters found ({len(valid_clusters)})")
        return image.copy(), 0.3, []
    
    # Draw the clusters with different colors
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)]
    
    detected_regions = []
    
    # Draw each cluster
    for i, (label, cluster_points) in enumerate(valid_clusters.items()):
        color = colors[i % len(colors)]
        
        # Convert points to int32
        pts = np.array(cluster_points, dtype=np.int32)
        
        # Get min bounding rectangle
        x, y, w, h = cv2.boundingRect(pts)
        
        # Adjust rectangle to block size
        x = (x // block_size) * block_size
        y = (y // block_size) * block_size
        w = ((w + block_size - 1) // block_size) * block_size
        h = ((h + block_size - 1) // block_size) * block_size
        
        # Draw rectangle
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 3)
        
        # Store detected region
        detected_regions.append({
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h)
        })
    
    # Calculate confidence based on number of clusters and matches
    # Improved confidence calculation for more accurate results
    n_valid_clusters = len(valid_clusters)
    n_similar_pairs = len(similar_pairs)
    
    # Base confidence on multiple factors
    cluster_factor = min(1.0, (n_valid_clusters - 1) * 0.2)  # More clusters = higher confidence
    pairs_factor = min(1.0, n_similar_pairs / 50)  # More similar pairs = higher confidence
    
    # Combine factors with weights
    confidence = min(0.95, 0.3 + (cluster_factor * 0.4) + (pairs_factor * 0.6))
    
    elapsed_time = time.time() - start_time
    logger.info(f"Copy-move DCT detection completed in {elapsed_time:.2f} seconds with confidence {confidence:.2f}")
    
    return Image.fromarray(vis_img), confidence, detected_regions 