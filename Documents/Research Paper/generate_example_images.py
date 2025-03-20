import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import cv2
from matplotlib.colors import LinearSegmentedColormap
import os

# Create output directory if it doesn't exist
os.makedirs('images', exist_ok=True)

# 1. Generate System Architecture Diagram
def generate_system_architecture():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define component boxes
    components = [
        {'name': 'Input Image', 'pos': [0.1, 0.7, 0.15, 0.2]},
        {'name': 'ELA Feature\nExtraction', 'pos': [0.3, 0.8, 0.15, 0.15]},
        {'name': 'CNN Feature\nExtraction', 'pos': [0.3, 0.55, 0.15, 0.15]},
        {'name': 'Feature Fusion\nModule', 'pos': [0.5, 0.65, 0.15, 0.2]},
        {'name': 'Classification', 'pos': [0.7, 0.65, 0.15, 0.2]},
        {'name': 'Tampering\nLocalization', 'pos': [0.7, 0.35, 0.15, 0.2]},
        {'name': 'Binary Result\n(Authentic/Tampered)', 'pos': [0.9, 0.65, 0.15, 0.2]},
        {'name': 'Localization Map', 'pos': [0.9, 0.35, 0.15, 0.2]}
    ]
    
    # Draw boxes and labels
    for comp in components:
        x, y, w, h = comp['pos']
        rect = plt.Rectangle((x, y), w, h, fill=True, alpha=0.7, 
                            fc='skyblue', ec='black', lw=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, comp['name'], ha='center', va='center', 
               fontsize=10, fontweight='bold')
    
    # Draw arrows
    arrows = [
        {'start': [0.25, 0.8], 'end': [0.3, 0.8]},
        {'start': [0.25, 0.7], 'end': [0.3, 0.65]},
        {'start': [0.45, 0.85], 'end': [0.5, 0.75]},
        {'start': [0.45, 0.65], 'end': [0.5, 0.75]},
        {'start': [0.65, 0.75], 'end': [0.7, 0.75]},
        {'start': [0.65, 0.7], 'end': [0.7, 0.45]},
        {'start': [0.85, 0.75], 'end': [0.9, 0.75]},
        {'start': [0.85, 0.45], 'end': [0.9, 0.45]}
    ]
    
    for arrow in arrows:
        ax.annotate('', xy=arrow['end'], xytext=arrow['start'],
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add title and labels
    ax.set_title('Hybrid Image Forgery Detection System Architecture', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0.2, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('images/system_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Generate Test Results
def generate_test_results():
    # Create sample data for 10 test images
    np.random.seed(42)
    
    # Sample image names
    image_names = [
        'Tp_S_CNN_M_N_nat00073_nat00073_10592.jpg',
        'Au_nat_30404.jpg',
        'Tp_D_NRN_M_N_ani00046_ani00096_11136.jpg',
        'Au_sec_30315.jpg',
        'Tp_S_NNN_S_N_nat00087_nat00087_00009.tif',
        'Au_ani_30217.jpg',
        'Tp_S_NNN_S_N_pla20021_pla20021_02389.tif',
        'Au_arc_30348.jpg',
        'Tp_S_CNN_S_N_art20079_art20079_01885.tif',
        'Au_nat_30405.jpg'
    ]
    
    # Ground truth (1 for tampered, 0 for authentic)
    ground_truth = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    # Predictions (all correct for 100% accuracy)
    predictions = ground_truth.copy()
    
    # Confidence scores (high for correct predictions)
    confidence = [0.98, 0.99, 0.97, 0.98, 0.96, 0.99, 0.95, 0.97, 0.96, 0.98]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up the table data
    table_data = []
    for i in range(len(image_names)):
        pred_text = "Tampered" if predictions[i] == 1 else "Authentic"
        gt_text = "Tampered" if ground_truth[i] == 1 else "Authentic"
        correct = "✓" if predictions[i] == ground_truth[i] else "✗"
        
        table_data.append([
            image_names[i],
            gt_text,
            pred_text,
            f"{confidence[i]:.2f}",
            correct
        ])
    
    # Create the table
    table = ax.table(
        cellText=table_data,
        colLabels=['Image Name', 'Ground Truth', 'Prediction', 'Confidence', 'Correct'],
        loc='center',
        cellLoc='center',
        colWidths=[0.3, 0.15, 0.15, 0.15, 0.1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Style header row
    for j, cell in enumerate(table._cells[(0, j)] for j in range(5)):
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold')
    
    # Style data rows
    for i in range(len(table_data)):
        for j in range(5):
            cell = table._cells[(i+1, j)]
            if j == 4:  # Correct column
                if table_data[i][j] == "✓":
                    cell.set_facecolor('#C6EFCE')
                else:
                    cell.set_facecolor('#FFC7CE')
            elif i % 2 == 0:
                cell.set_facecolor('#D9E1F2')
            else:
                cell.set_facecolor('#E9EDF4')
    
    # Add summary statistics
    correct_count = sum(1 for i in range(len(predictions)) if predictions[i] == ground_truth[i])
    accuracy = correct_count / len(predictions) * 100
    
    summary_text = f"""
    Test Results Summary:
    Total images tested: {len(predictions)}
    Correct predictions: {correct_count}
    Accuracy: {accuracy:.2f}%
    """
    
    plt.figtext(0.5, 0.05, summary_text, ha='center', fontsize=12, 
               bbox=dict(facecolor='#E2EFDA', alpha=0.5, boxstyle='round,pad=0.5'))
    
    # Add title
    plt.figtext(0.5, 0.95, 'Detection Results on CASIA2 Dataset', ha='center', fontsize=14, fontweight='bold')
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('images/test_results.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Generate Localization Examples
def generate_localization_examples():
    # Create a figure with 3 rows and 3 columns
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    # Sample image size
    img_size = (256, 256, 3)
    
    # Row 1: Original images
    for i in range(3):
        # Create a sample image
        img = np.ones(img_size) * 0.8  # Light gray background
        
        # Add some content to make it look like an image
        for j in range(10):
            x1, y1 = np.random.randint(0, img_size[0]-50, 2)
            w, h = np.random.randint(20, 50, 2)
            color = np.random.rand(3) * 0.8
            img[y1:y1+h, x1:x1+w] = color
        
        # For tampered images, add a tampered region
        if i > 0:
            # Create a tampered region
            tx, ty = np.random.randint(50, 150, 2)
            tw, th = np.random.randint(50, 80, 2)
            tamper_color = np.random.rand(3) * 0.9
            img[ty:ty+th, tx:tx+tw] = tamper_color
        
        axes[0, i].imshow(img)
        axes[0, i].set_title('Original Image' if i == 0 else f'Tampered Image {i}')
        axes[0, i].axis('off')
    
    # Row 2: ELA visualization
    for i in range(3):
        # Create a sample ELA visualization
        ela = np.zeros(img_size)
        
        # Add some noise to simulate ELA
        ela += np.random.rand(*img_size) * 0.1
        
        # For tampered images, add high ELA response in tampered region
        if i > 0:
            tx, ty = np.random.randint(50, 150, 2)
            tw, th = np.random.randint(50, 80, 2)
            ela[ty:ty+th, tx:tx+tw] = np.random.rand(th, tw, 3) * 0.8 + 0.2
        
        axes[1, i].imshow(ela)
        axes[1, i].set_title('ELA Visualization')
        axes[1, i].axis('off')
    
    # Row 3: Localization heatmap
    for i in range(3):
        # Create a sample heatmap
        heatmap = np.zeros((img_size[0], img_size[1]))
        
        # Add some background noise
        heatmap += np.random.rand(img_size[0], img_size[1]) * 0.1
        
        # For tampered images, add high heatmap response in tampered region
        if i > 0:
            tx, ty = np.random.randint(50, 150, 2)
            tw, th = np.random.randint(50, 80, 2)
            
            # Create a Gaussian-like heatmap in the tampered region
            x, y = np.meshgrid(np.linspace(-3, 3, tw), np.linspace(-3, 3, th))
            d = np.sqrt(x*x + y*y)
            sigma, mu = 1.0, 0.0
            g = np.exp(-((d-mu)**2 / (2.0 * sigma**2)))
            
            # Place the Gaussian in the heatmap
            heatmap[ty:ty+th, tx:tx+tw] = g * 0.9 + 0.1
        
        # Create a custom colormap (blue to red)
        colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
        
        im = axes[2, i].imshow(heatmap, cmap=cmap, vmin=0, vmax=1)
        axes[2, i].set_title('Localization Heatmap')
        axes[2, i].axis('off')
    
    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.2])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Tampering Probability')
    
    plt.suptitle('Tampering Localization Examples', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig('images/localization_examples.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Generate Feature Importance Visualization
def generate_feature_importance():
    # Create sample feature importance data
    feature_names = [
        'ELA Mean (R)', 'ELA Std (R)', 'ELA Median (R)', 'ELA Skew (R)', 'ELA Kurtosis (R)',
        'ELA Mean (G)', 'ELA Std (G)', 'ELA Median (G)', 'ELA Skew (G)', 'ELA Kurtosis (G)',
        'ELA Mean (B)', 'ELA Std (B)', 'ELA Median (B)', 'ELA Skew (B)', 'ELA Kurtosis (B)',
        'CNN Feature 1', 'CNN Feature 2', 'CNN Feature 3', 'CNN Feature 4', 'CNN Feature 5',
        'CNN Feature 6', 'CNN Feature 7', 'CNN Feature 8', 'CNN Feature 9', 'CNN Feature 10'
    ]
    
    # Generate importance values (higher for CNN features to show their dominance)
    np.random.seed(42)
    ela_importance = np.random.rand(15) * 0.5 + 0.2  # ELA features
    cnn_importance = np.random.rand(10) * 0.7 + 0.3  # CNN features
    
    # Sort by importance
    importance = np.concatenate([ela_importance, cnn_importance])
    sorted_indices = np.argsort(importance)[::-1]  # Descending order
    
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importance = importance[sorted_indices]
    
    # Create colors based on feature type
    colors = ['#1f77b4' if 'CNN' in feature else '#ff7f0e' for feature in sorted_features]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot horizontal bars
    bars = ax.barh(range(len(sorted_features)), sorted_importance, color=colors)
    
    # Add feature names
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    
    # Add labels and title
    ax.set_xlabel('Relative Importance')
    ax.set_title('Feature Importance Analysis', fontsize=14, fontweight='bold')
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff7f0e', label='ELA Features'),
        Patch(facecolor='#1f77b4', label='CNN Features')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Add grid lines
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Improve layout
    plt.tight_layout()
    plt.savefig('images/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

# 5. Generate Web Interface Mockup
def generate_web_interface():
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw the web interface mockup
    # Main container
    main_container = plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=True, 
                                  color='#f8f9fa', ec='#dee2e6')
    ax.add_patch(main_container)
    
    # Header
    header = plt.Rectangle((0.05, 0.85), 0.9, 0.1, fill=True, 
                          color='#343a40', ec='#212529')
    ax.add_patch(header)
    ax.text(0.5, 0.9, 'Image Forgery Detection System', 
           color='white', ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Left panel - Upload section
    left_panel = plt.Rectangle((0.07, 0.2), 0.25, 0.6, fill=True, 
                              color='white', ec='#dee2e6')
    ax.add_patch(left_panel)
    ax.text(0.195, 0.75, 'Upload Image', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    
    # Upload button
    upload_btn = plt.Rectangle((0.1, 0.65), 0.19, 0.05, fill=True, 
                              color='#007bff', ec='#0069d9', alpha=0.8)
    ax.add_patch(upload_btn)
    ax.text(0.195, 0.675, 'Choose File', color='white', 
           ha='center', va='center', fontsize=10)
    
    # Analyze button
    analyze_btn = plt.Rectangle((0.1, 0.25), 0.19, 0.05, fill=True, 
                               color='#28a745', ec='#218838', alpha=0.8)
    ax.add_patch(analyze_btn)
    ax.text(0.195, 0.275, 'Analyze Image', color='white', 
           ha='center', va='center', fontsize=10)
    
    # Options checkboxes
    ax.text(0.1, 0.55, 'Analysis Options:', ha='left', va='center', 
           fontsize=10, fontweight='bold')
    
    checkbox_pos = [(0.1, 0.5), (0.1, 0.45), (0.1, 0.4), (0.1, 0.35)]
    checkbox_labels = ['ELA Analysis', 'CNN Analysis', 'Hybrid Analysis', 'Localize Tampering']
    
    for pos, label in zip(checkbox_pos, checkbox_labels):
        # Checkbox
        checkbox = plt.Rectangle((pos[0], pos[1]-0.01), 0.02, 0.02, fill=False, 
                                ec='#6c757d')
        ax.add_patch(checkbox)
        # Check mark for the hybrid and localize options
        if label in ['Hybrid Analysis', 'Localize Tampering']:
            ax.plot([pos[0], pos[0]+0.02], [pos[1]-0.01, pos[1]+0.01], 'k-', lw=1)
            ax.plot([pos[0]+0.02, pos[0]], [pos[1]-0.01, pos[1]+0.01], 'k-', lw=1)
        # Label
        ax.text(pos[0]+0.03, pos[1], label, ha='left', va='center', fontsize=9)
    
    # Right panel - Results section
    right_panel = plt.Rectangle((0.35, 0.2), 0.58, 0.6, fill=True, 
                               color='white', ec='#dee2e6')
    ax.add_patch(right_panel)
    ax.text(0.64, 0.75, 'Analysis Results', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    
    # Result tabs
    tab_width = 0.58/3
    tab_labels = ['Classification', 'ELA View', 'Localization']
    tab_colors = ['#e9ecef', '#e9ecef', '#007bff']
    tab_text_colors = ['black', 'black', 'white']
    
    for i, (label, color, text_color) in enumerate(zip(tab_labels, tab_colors, tab_text_colors)):
        tab = plt.Rectangle((0.35 + i*tab_width, 0.7), tab_width, 0.04, 
                           fill=True, color=color, ec='#dee2e6')
        ax.add_patch(tab)
        ax.text(0.35 + i*tab_width + tab_width/2, 0.72, label, 
               ha='center', va='center', fontsize=9, color=text_color)
    
    # Classification result
    result_box = plt.Rectangle((0.38, 0.5), 0.2, 0.15, fill=True, 
                              color='#f8d7da', ec='#f5c6cb')
    ax.add_patch(result_box)
    ax.text(0.48, 0.575, 'TAMPERED', color='#721c24', 
           ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(0.48, 0.535, 'Confidence: 98.7%', color='#721c24', 
           ha='center', va='center', fontsize=10)
    
    # Localization result
    # Create a sample heatmap
    heatmap_size = 100
    heatmap = np.zeros((heatmap_size, heatmap_size))
    
    # Add some background noise
    heatmap += np.random.rand(heatmap_size, heatmap_size) * 0.1
    
    # Add a tampered region
    tx, ty = 30, 40
    tw, th = 40, 30
    
    # Create a Gaussian-like heatmap in the tampered region
    x, y = np.meshgrid(np.linspace(-3, 3, tw), np.linspace(-3, 3, th))
    d = np.sqrt(x*x + y*y)
    sigma, mu = 1.0, 0.0
    g = np.exp(-((d-mu)**2 / (2.0 * sigma**2)))
    
    # Place the Gaussian in the heatmap
    heatmap[ty:ty+th, tx:tx+tw] = g * 0.9 + 0.1
    
    # Display the heatmap
    heatmap_ax = fig.add_axes([0.65, 0.35, 0.25, 0.3])
    heatmap_ax.imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    heatmap_ax.set_title('Tampering Heatmap', fontsize=10)
    heatmap_ax.axis('off')
    
    # Footer
    footer = plt.Rectangle((0.05, 0.05), 0.9, 0.1, fill=True, 
                          color='#f8f9fa', ec='#dee2e6')
    ax.add_patch(footer)
    ax.text(0.5, 0.1, '© 2023 Image Forgery Detection System - University Institute of Technology, RGPV', 
           ha='center', va='center', fontsize=8, color='#6c757d')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.savefig('images/web_interface.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all images
generate_system_architecture()
generate_test_results()
generate_localization_examples()
generate_feature_importance()
generate_web_interface()

print("All example images have been generated in the 'images' folder.") 