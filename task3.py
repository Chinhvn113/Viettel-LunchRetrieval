import cv2
import numpy as np
import torch
import open3d as o3d
from PIL import Image
from scipy import ndimage
import os
import csv  # ### THAY ĐỔI ###: Thêm thư viện csv để ghi file
import zipfile
# SAM 2.1 Imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from lean1 import find_surface_normal


# ==================== CONFIGURATION ====================

# ### THAY ĐỔI ###: Chuyển từ đường dẫn file sang đường dẫn thư mục
# --- Folder Paths ---
RGB_FOLDER = "/mlcv2/WorkingSpace/Personal/chinhnm/LunchBox/Viettel/Data/ThiSinh/rgb"
DEPTH_FOLDER = "/mlcv2/WorkingSpace/Personal/chinhnm/LunchBox/Viettel/Data/ThiSinh/depth"
OUTPUT_FOLDER = "/mlcv2/WorkingSpace/Personal/chinhnm/LunchBox/Viettel/visualize_output_private" # Thư mục chứa ảnh và file PLY
CSV_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, "Submission_3D.csv") # Đường dẫn file CSV kết quả

# --- SAM Checkpoint ---
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Camera Intrinsics (Depth Camera) ---
depth_fx = 650.0616455078125
depth_fy = 650.0616455078125
depth_cx = 649.5928055078125
depth_cy = 360.9415588378906

# Color intrinsics (nếu cần transform sang color frame)
color_fx = 643.90087890625
color_fy = 643.1365356445312
color_cx = 650.2113037109375
color_cy = 355.79550326171875

R_depth_to_color = np.array([
    [0.9999898076057434, -0.00020347206736914814, -0.004507721401751041],
    [0.00018898719281423837, 0.9999948143959045, -0.0032135415822267532],
    [0.004508351907134056, 0.003212657058611512, 0.9999846816062927]
])
t_depth_to_color = np.array([-0.05905, 8.67399e-5, 0.00041])

# --- Processing Parameters ---
ROI_X, ROI_Y, ROI_W, ROI_H = 560, 150, 300, 330
MAX_MASK_AREA_RATIO = 0.65
MIN_FILL_RATIO = 0.9
# ==================== ADVANCED PARAMETERS ====================
EDGE_MARGIN = 10
MIN_DEPTH_THRESHOLD = 300
MAX_DEPTH_THRESHOLD = 5000
MEDIAN_FILTER_SIZE = 1
CLUSTER_SIZE = 1
HIGH_CONFIDENCE_THRESHOLD = 0.97
NORMAL_VIS_LENGTH = 0.1
FILL_RATIO_IMPROVEMENT_THRESHOLD = 0.1
POISSON_DEPTH = 9
# =================================================================================
# CÁC HÀM PHỤ (giữ nguyên không thay đổi)
# =================================================================================
def calculate_mask_centroid(mask):
    """
    Calculates the center of mass (centroid) of a binary mask.
    Returns (cx, cy) or None if the mask is empty.
    """
    mask_uint8 = mask.astype(np.uint8)
    M = cv2.moments(mask_uint8)

    # Check if the mask area is zero to avoid division by zero
    if M["m00"] == 0:
        print("Warning: Cannot calculate centroid of an empty mask.")
        return None

    # Calculate x, y coordinate of center
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return (cx, cy)

def check_mask_similarity(mask1, mask2, overlap_thresh=0.9, area_sim_thresh=0.9):
    """
    Checks if two masks are similar based on overlap and area similarity.
    This is a much more robust method than using centroids.
    """
    area1 = np.sum(mask1)
    area2 = np.sum(mask2)
    
    if area1 == 0 or area2 == 0:
        return False
        
    # 1. Check for area similarity
    min_area = min(area1, area2)
    max_area = max(area1, area2)
    if min_area / max_area < area_sim_thresh:
        return False # They are not similar enough in size

    # 2. Check for high overlap relative to the smaller mask
    intersection = np.sum(np.logical_and(mask1, mask2))
    if intersection / min_area < overlap_thresh:
        return False # They don't overlap enough
        
    # If both checks pass, they are considered the same mask
    return True

def calculate_fill_ratio(mask):
    """
    Calculates the ratio of the mask's area to its oriented bounding box's area.
    A high value indicates a compact, solid shape.
    """
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0

    largest_contour = max(contours, key=cv2.contourArea)
    mask_area = cv2.contourArea(largest_contour)

    if mask_area == 0:
        return 0.0

    rect = cv2.minAreaRect(largest_contour)
    # rect[1] is a tuple of (width, height)
    bbox_area = rect[1][0] * rect[1][1]

    if bbox_area == 0:
        return 0.0

    return mask_area / bbox_area
def find_closest_point_in_roi(depth_image, roi_x, roi_y, roi_w, roi_h,
                               edge_margin=EDGE_MARGIN,
                               min_depth=MIN_DEPTH_THRESHOLD,
                               max_depth=MAX_DEPTH_THRESHOLD,
                               median_kernel=MEDIAN_FILTER_SIZE,
                               cluster_size=CLUSTER_SIZE,
                               verbose=True):
    # ... (Nội dung hàm này giữ nguyên)
    # 1. Crop ROI từ depth image
    roi_depth = depth_image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
    
    if verbose:
        print(f"\n=== PROCESSING ROI ({roi_x}, {roi_y}, {roi_w}, {roi_h}) ===")
        print(f"Original ROI shape: {roi_depth.shape}")
    
    # 2. Apply median filter để giảm noise
    if median_kernel > 1:
        roi_depth_filtered = cv2.medianBlur(roi_depth, median_kernel)
        if verbose:
            print(f"Applied median filter (kernel={median_kernel}x{median_kernel})")
    else:
        roi_depth_filtered = roi_depth.copy()
    
    # 3. Tạo valid mask
    valid_mask = (roi_depth_filtered > min_depth) & (roi_depth_filtered < max_depth)
    
    # 4. Loại bỏ edge pixels
    if edge_margin > 0:
        edge_mask = np.ones_like(valid_mask, dtype=bool)
        edge_mask[:edge_margin, :] = False
        edge_mask[-edge_margin:, :] = False
        edge_mask[:, :edge_margin] = False
        edge_mask[:, -edge_margin:] = False
        valid_mask = valid_mask & edge_mask
        
        if verbose:
            print(f"Removed {edge_margin}px edges")
    
    if not np.any(valid_mask):
        raise ValueError("No valid depth values in ROI after filtering!")
    
    valid_count = np.sum(valid_mask)
    if verbose:
        print(f"Valid pixels: {valid_count}/{roi_depth.size} ({100*valid_count/roi_depth.size:.1f}%)")
    
    # 5. Tìm điểm/vùng có depth nhỏ nhất
    if cluster_size > 1:
        # Tìm vùng cluster_size x cluster_size có depth trung bình nhỏ nhất
        roi_depth_masked = roi_depth_filtered.copy().astype(float)
        roi_depth_masked[~valid_mask] = np.nan
        
        # Compute local mean using uniform filter
        kernel = np.ones((cluster_size, cluster_size)) / (cluster_size * cluster_size)
        local_mean = ndimage.uniform_filter(roi_depth_masked, size=cluster_size, mode='constant', cval=np.nan)
        
        # Find minimum of local means
        valid_local_mean = np.where(valid_mask, local_mean, np.inf)
        min_idx = np.argmin(valid_local_mean)
        local_y, local_x = np.unravel_index(min_idx, roi_depth.shape)
        
        if verbose:
            print(f"Using cluster averaging (size={cluster_size}x{cluster_size})")
            print(f"Local mean depth at closest point: {local_mean[local_y, local_x]:.1f}mm")
    else:
        # Tìm pixel đơn lẻ có depth nhỏ nhất
        roi_depth_masked = roi_depth_filtered.copy()
        roi_depth_masked[~valid_mask] = max_depth + 1
        
        min_idx = np.argmin(roi_depth_masked)
        local_y, local_x = np.unravel_index(min_idx, roi_depth.shape)
    
    # 6. Chuyển về tọa độ trong ảnh gốc
    pixel_x = roi_x + local_x
    pixel_y = roi_y + local_y
    depth_value = depth_image[pixel_y, pixel_x]
    
    # Check if found point is at edge (shouldn't happen after filtering)
    is_at_edge = (local_x < edge_margin or local_x >= roi_w - edge_margin or
                  local_y < edge_margin or local_y >= roi_h - edge_margin)
    
    if verbose:
        print(f"\n=== RESULT ===")
        print(f"Closest point pixel: ({pixel_x}, {pixel_y})")
        print(f"Position in ROI: ({local_x}, {local_y})")
        print(f"Depth value: {depth_value} mm")
        if is_at_edge:
            print("⚠️  WARNING: Point is at ROI edge!")
    
    # 7. Unproject sang 3D (sử dụng depth intrinsics)
    Z = depth_value / 1000.0  # Convert mm to meters
    X = (pixel_x - depth_cx) * Z / depth_fx
    Y = (pixel_y - depth_cy) * Z / depth_fy
    
    point_3d = (X, Y, Z)
    
    if verbose:
        print(f"3D coordinates: X={X:.4f}m, Y={Y:.4f}m, Z={Z:.4f}m")
    
    return pixel_x, pixel_y, depth_value, point_3d
def find_normal_from_obb(pcd):
    """
    Calculates the surface normal by finding the top face of the object's
    Oriented Bounding Box (OBB).
    """
    print("\n--- Finding Normal Vector from 3D Oriented Bounding Box ---")
    if not pcd.has_points():
        print("Warning: Point cloud is empty. Cannot calculate normal from OBB.")
        return None

    try:
        # 1. Get the Oriented Bounding Box
        obb = pcd.get_oriented_bounding_box()
        
        # 2. The obb.R rotation matrix's columns are the principal axes of the box.
        box_axes = obb.R
        
        # 3. Find the axis that is most aligned with the camera's Z-axis.
        #    This will be the normal vector of the top/bottom surfaces.
        #    We check the absolute value of the Z component of each axis vector.
        z_components_abs = np.abs(box_axes[2, :])
        top_surface_axis_index = np.argmax(z_components_abs)
        
        # 4. Get the vector for this axis.
        normal_vector = box_axes[:, top_surface_axis_index]
        
        # 5. Ensure the normal points towards the camera (negative Z direction).
        #    The top surface's normal should have a negative Z value in camera coordinates.
        if normal_vector[2] > 0:
            normal_vector = -normal_vector
            
        print(f"OBB main axes identified. Chosen normal vector: {normal_vector}")
        return normal_vector

    except Exception as e:
        print(f"Error calculating normal from OBB: {e}")
        return None

# ### THAY ĐỔI ###: Thêm hàm helper để tính IoU
def calculate_containment(mask1, mask2):
    """
    Calculates how much of the smaller mask is contained within the larger one.
    Returns: Intersection / Area(smaller_mask)
    """
    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)
    
    intersection = np.logical_and(m1, m2).sum()
    area1 = m1.sum()
    area2 = m2.sum()

    if area1 == 0 or area2 == 0:
        return 0.0

    smaller_area = min(area1, area2)
    
    # If the smaller mask has zero area, containment is ill-defined.
    if smaller_area == 0:
        return 1.0 if intersection > 0 else 0.0

    return intersection / smaller_area
def calculate_iou(mask1, mask2):
    """Calculates Intersection over Union."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0
# ### THAY ĐỔI ###: Logic chọn mask đã được cập nhật
def segment_object_with_sam(predictor, rgb_image, point_prompt, roi_w, roi_h):
    """
    Segments an object. If filtering removes all masks, it falls back to the original list.
    """
    print("\n--- Part 2: Segmenting object with SAM (Size -> Shape -> Containment) ---")
    
    predictor.set_image(rgb_image) 
    input_point = point_prompt[np.newaxis, :]
    input_label = np.array([1])
    
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    
    if len(masks) == 0:
        print("Warning: SAM did not return any masks.")
        return np.zeros(rgb_image.shape[:2], dtype=bool), [], []
    # --- High-Confidence Override ---
    max_score = np.max(scores)
    if max_score >= HIGH_CONFIDENCE_THRESHOLD:
        print(f"\n--- High Confidence Override ---")
        print(f"Detected a very high score of {max_score:.4f} (>= {HIGH_CONFIDENCE_THRESHOLD}). Bypassing filters.")
        best_mask_idx = np.argmax(scores)
        final_mask = masks[best_mask_idx]
        print(f"Directly selecting mask {best_mask_idx} as the final choice.")
        return final_mask.astype(bool), masks, scores

    print(f"\n--- No high-confidence mask found (max score: {max_score:.4f}). Proceeding with full filtering pipeline. ---")
    print("\n--- Checking for duplicate masks (similar area and position) ---")
    processed_indices = set()
    duplicate_groups = []
    for i in range(len(masks)):
        if i in processed_indices: continue
        current_group = [i]
        processed_indices.add(i)
        for j in range(i + 1, len(masks)):
            if j in processed_indices: continue
            if check_mask_similarity(masks[i], masks[j]):
                current_group.append(j)
                processed_indices.add(j)
        if len(current_group) > 1:
            duplicate_groups.append(current_group)

    if duplicate_groups:
        print(f"Found {len(duplicate_groups)} group(s) of duplicate masks. Prioritizing this logic.")
        best_duplicate_group = duplicate_groups[0]
        print(f"Processing first duplicate group: {best_duplicate_group}")

        best_mask_in_group = None
        best_fill_ratio = -1.0
        best_mask_idx = -1
        for idx in best_duplicate_group:
            mask = masks[idx]
            fill_ratio = calculate_fill_ratio(mask)
            print(f"  - Mask index {idx} (score: {scores[idx]:.4f}) has fill ratio: {fill_ratio:.4f}")
            if fill_ratio > best_fill_ratio:
                best_fill_ratio = fill_ratio
                best_mask_in_group = mask
                best_mask_idx = idx
        
        print(f"Selected mask {best_mask_idx} from duplicates based on best fill ratio ({best_fill_ratio:.4f}).")
        return best_mask_in_group.astype(bool), masks, scores
    # --- 1. & 2. Filter masks by Size and Shape ---
    # 3a. Size Filter (Hard Cutoff)
    max_allowed_area = roi_w * roi_h * MAX_MASK_AREA_RATIO
    print(f"--- Filtering by Size (max area: {max_allowed_area:.0f}) ---")
    size_filtered_candidates = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask_area = np.sum(mask)
        if mask_area < max_allowed_area:
            fill_ratio = calculate_fill_ratio(mask)
            size_filtered_candidates.append({
                'mask': mask, 'score': score, 'original_index': i, 'fill_ratio': fill_ratio
            })
        else:
            print(f"  -> Rejecting mask {i} - Too large.")

    if not size_filtered_candidates:
        print("Warning: All masks were filtered out for being too large. No valid candidates remain.")
        return np.zeros(rgb_image.shape[:2], dtype=bool), masks, scores

    # 3b. Shape Filter
    print(f"--- Filtering {len(size_filtered_candidates)} candidates by Shape (min fill ratio: {MIN_FILL_RATIO}) ---")
    shape_filtered_candidates = []
    for candidate in size_filtered_candidates:
        if candidate['fill_ratio'] >= MIN_FILL_RATIO:
            shape_filtered_candidates.append(candidate)
        else:
            print(f"  -> Rejecting mask {candidate['original_index']} - Poor shape (fill ratio: {candidate['fill_ratio']:.2f}).")

    # 3c. Final Selection Logic with New Fallback
    final_mask = None
    final_candidate_info = None
    
    # Determine which list of candidates to use for the final containment check
    if shape_filtered_candidates:
        # Ideal Case: We have candidates that passed all filters.
        print(f"\nFound {len(shape_filtered_candidates)} candidates passing all filters. Refining with containment logic.")
        candidates_for_refinement = shape_filtered_candidates
    elif size_filtered_candidates:
        # Fallback Case: No masks passed the shape test. Use the size-appropriate list.
        print("\n!!! WARNING: No masks passed shape filter. Falling back to containment logic on all size-appropriate masks. !!!")
        candidates_for_refinement = size_filtered_candidates
    else:
        # Should not be reached, but included for safety.
        print("Error: No valid masks found after all filtering stages.")
        return np.zeros(rgb_image.shape[:2], dtype=bool), masks, scores

    # Run the containment logic on the chosen list of candidates
    print(f"\n--- Final Refinement (Hierarchical: Shape -> Containment) ---")
    candidates_for_refinement.sort(key=lambda x: x['score'], reverse=True)
    
    best_so_far_candidate = candidates_for_refinement[0]
    print(f"Starting with best candidate (by score): Index {best_so_far_candidate['original_index']} (Fill Ratio: {best_so_far_candidate['fill_ratio']:.4f})")

    # Iterate through the rest of the candidates to challenge the current best
    for i in range(1, len(candidates_for_refinement)):
        challenger_candidate = candidates_for_refinement[i]
        
        # --- LOGIC 1: Significant Shape (Fill Ratio) Improvement ---
        # Does the challenger have a substantially better shape?
        if challenger_candidate['fill_ratio'] > best_so_far_candidate['fill_ratio'] + FILL_RATIO_IMPROVEMENT_THRESHOLD:
            print(f"  - Challenger [Idx {challenger_candidate['original_index']}]'s fill ratio ({challenger_candidate['fill_ratio']:.4f}) is "
                  f"significantly better than current best [Idx {best_so_far_candidate['original_index']}] ({best_so_far_candidate['fill_ratio']:.4f}).")
            print(f"  --> SWITCHING best candidate.")
            best_so_far_candidate = challenger_candidate
            continue # Move to the next challenger with our new champion

        # --- LOGIC 2: Containment (Fallback for similar shapes) ---
        # If shapes are similar, check if the challenger is a larger, more complete version of the current best.
        best_mask = best_so_far_candidate['mask']
        challenger_mask = challenger_candidate['mask']
        
        # We only care about the case where the challenger is LARGER and contains the current best
        if np.sum(challenger_mask) > np.sum(best_mask):
            containment_score = calculate_containment(best_mask, challenger_mask)
            if containment_score > 0.95:
                print(f"  - Challenger [Idx {challenger_candidate['original_index']}] contains the current best [Idx {best_so_far_candidate['original_index']}] (Containment: {containment_score:.2f}) and is larger.")
                print(f"  --> SWITCHING best candidate.")
                best_so_far_candidate = challenger_candidate
        # else:
        #     print(f"  - Challenger [Idx {challenger_candidate['original_index']}] does not meet criteria to replace current best.")

    final_candidate_info = best_so_far_candidate
    final_mask = final_candidate_info['mask']
    
    print(f"\nSegmentation complete. Final selected mask original index: {final_candidate_info['original_index']}")
    return final_mask.astype(bool), masks, scores

def segment_object_with_sam_old(predictor, rgb_image, point_prompt, roi_w, roi_h):
    """
    Segments an object using SAM with filtering by Size, Shape, and Confidence Score.
    """
    print("\n--- Part 2: Segmenting object with SAM (Size -> Shape -> Score) ---")
    
    predictor.set_image(rgb_image) 
    input_point = point_prompt[np.newaxis, :]
    input_label = np.array([1])
    
    mask = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    
    return mask[0][0].astype(bool)


# ... (Các hàm extract_point_cloud_from_mask, get_oriented_bbox_2d, visualize_results giữ nguyên) ...
def extract_point_cloud_from_mask(depth_image, rgb_image, mask, intrinsics):
    print("\n=== Part 3: Extracting point cloud from mask ===")
    masked_depth = depth_image.copy()
    masked_depth[~mask] = 0
    o3d_color = o3d.geometry.Image(rgb_image) 
    o3d_depth = o3d.geometry.Image(masked_depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth, 
        depth_scale=1000.0,
        depth_trunc=MAX_DEPTH_THRESHOLD / 1000.0,
        convert_rgb_to_intensity=False
    )
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=intrinsics['width'],
        height=intrinsics['height'],
        fx=intrinsics['fx'],
        fy=intrinsics['fy'],
        cx=intrinsics['cx'],
        cy=intrinsics['cy']
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d_intrinsics
    )
    print(f"Extracted point cloud with {len(pcd.points)} points.")
    return pcd

def get_oriented_bbox_2d(mask):
    print("\n=== Part 4: Finding 2D oriented bounding box ===")
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Warning: No contours found in the mask.")
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    print(f"Found BBox with center ({rect[0][0]:.1f}, {rect[0][1]:.1f}) and angle {rect[2]:.1f}°")
    return rect
def get_robust_depth_from_mask(depth_image, mask, min_depth=MIN_DEPTH_THRESHOLD):
    """
    Calculates a robust depth value for an object using the median of valid depth points
    within its segmentation mask. This is used as a fallback.
    """
    object_depth_values = depth_image[mask]
    valid_depths = object_depth_values[object_depth_values > min_depth]
    
    if len(valid_depths) == 0:
        print(f"Warning: (Fallback) No valid depth points found within the mask.")
        return None
        
    median_depth = np.median(valid_depths)
    print(f"Success: (Fallback) Found {len(valid_depths)} valid points. Robust median depth: {median_depth:.1f} mm")
    return median_depth
def create_mesh_from_pointcloud(pcd, poisson_depth=POISSON_DEPTH):
    """
    Reconstructs a 3D mesh from a point cloud using Poisson surface reconstruction.
    """
    print("\n--- Part 3.5: Reconstructing 3D Mesh from Point Cloud ---")
    if not pcd.has_points() or len(pcd.points) < 100:
        print("Warning: Point cloud is too sparse for meshing. Skipping.")
        return None

    try:
        # 1. Estimate normals, which are required for Poisson reconstruction.
        print(f"Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(100) # Orient normals towards a consistent direction

        if not pcd.has_normals():
            print("Error: Normal estimation failed. Cannot create mesh.")
            return None

        # 2. Apply Poisson surface reconstruction.
        print(f"Applying Poisson surface reconstruction (depth={poisson_depth})...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson_depth, width=0, scale=1.1, linear_fit=False
        )
        
        # 3. Clean the mesh by removing low-density vertices (common artifacts).
        print("Cleaning mesh: Removing low-density vertices...")
        vertices_to_remove = densities < np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(vertices_to_remove)

        # 4. Keep only the largest connected component of the mesh.
        print("Cleaning mesh: Keeping only the largest cluster...")
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        
        if len(cluster_n_triangles) > 0:
            largest_cluster_idx = cluster_n_triangles.argmax()
            triangles_to_remove = triangle_clusters != largest_cluster_idx
            mesh.remove_triangles_by_mask(triangles_to_remove)
        
        mesh.remove_unreferenced_vertices() # Final cleanup
        print(f"Mesh reconstruction complete. Final mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")
        return mesh

    except Exception as e:
        print(f"An error occurred during mesh reconstruction: {e}")
        return None
def visualize_results(rgb_image, mask, point_prompt, rect, pose_3d, final_normal, intrinsics):
    """
    Visualizes the segmentation, points, bbox, and the projected 3D normal vector.
    """
    mask = mask.astype(bool)
    vis_image = rgb_image.copy()
    color_overlay = np.array([0, 255, 0], dtype=np.uint8)

    # Draw segmentation mask
    masked_pixels = vis_image[mask]
    blended_pixels = (masked_pixels * 0.5 + color_overlay * 0.5).astype(np.uint8)
    vis_image[mask] = blended_pixels

    # Draw prompt point
    cv2.circle(vis_image, tuple(point_prompt), 10, (255, 0, 0), -1, lineType=cv2.LINE_AA)
    cv2.circle(vis_image, tuple(point_prompt), 10, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    
    # Draw 2D oriented bounding box and its center
    if rect is not None:
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(vis_image, [box], 0, (0, 255, 255), 2, lineType=cv2.LINE_AA)
        center_2d = np.intp(rect[0])
        cv2.circle(vis_image, tuple(center_2d), 5, (255, 255, 0), -1, lineType=cv2.LINE_AA)

    # Draw the projected 3D normal vector
    if final_normal is not None and pose_3d is not None and rect is not None:
        # 1. Define start and end points in 3D space
        start_point_3d = pose_3d
        end_point_3d = start_point_3d + final_normal * NORMAL_VIS_LENGTH
        
        # 2. Project the 3D end point back to 2D image coordinates
        X_end, Y_end, Z_end = end_point_3d
        if Z_end > 0: # Check for valid depth
            x_end_2d = int((X_end / Z_end) * intrinsics['fx'] + intrinsics['cx'])
            y_end_2d = int((Y_end / Z_end) * intrinsics['fy'] + intrinsics['cy'])
            
            # 3. The start point is the 2D center of the object
            start_point_2d = tuple(np.intp(rect[0]))
            end_point_2d = (x_end_2d, y_end_2d)

            # 4. Draw a magenta arrow
            cv2.arrowedLine(vis_image, start_point_2d, end_point_2d, (255, 0, 255), 3, tipLength=0.3)
            
    return vis_image


# =================================================================================
# ### THAY ĐỔI ###: TÁCH LOGIC XỬ LÝ RA HÀM RIÊNG
# =================================================================================

def process_image_pair(rgb_path, depth_path, sam_predictor, output_vis_path, output_ply_path, output_mesh_path, all_masks_vis_folder):
    """
    Hàm này chứa toàn bộ pipeline xử lý cho MỘT cặp ảnh RGB và Depth.
    Nó sẽ trả về tọa độ trung tâm và vector pháp tuyến.
    """
    try:
        # --- Load Images ---
        print("--- Loading images ---")
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        rgb_image_bgr = cv2.imread(rgb_path)
        rgb_image = cv2.cvtColor(rgb_image_bgr, cv2.COLOR_BGR2RGB)
        
        if depth_image is None or rgb_image is None:
            print(f"Error: Could not load images: {rgb_path} or {depth_path}")
            return None, None
            
        H, W = depth_image.shape
        print(f"Image dimensions: {W}x{H}")

        # --- Part 1: Find the closest point in the ROI ---
        pixel_x, pixel_y, _, _ = find_closest_point_in_roi(
            depth_image, ROI_X, ROI_Y, ROI_W, ROI_H, verbose=False # verbose=False để log đỡ dài
        )
        point_prompt = np.array([pixel_x, pixel_y])

        # --- Part 2: Segment the object using SAM 2.1 ---
        segmentation_mask = segment_object_with_sam_old(
            sam_predictor, rgb_image, point_prompt, ROI_W, ROI_H
        )
        # ### THAY ĐỔI ###: Vòng lặp mới để lưu tất cả các mask
        # print(f"\n--- Saving all {len(all_masks)} generated mask visualizations ---")
        # for i, (mask, score) in enumerate(zip(all_masks, all_scores)):
        #     # Tạo visualization cho mask hiện tại
        #     # rect=None vì chúng ta chỉ quan tâm đến vùng mask, chưa có bbox
        #     vis_img = vis_img = visualize_results(rgb_image, mask, point_prompt, None, None, None, None)
            
        #     # Tạo tên file và đường dẫn
        #     vis_filename = f"mask_{i}_score_{score:.4f}.jpg"
        #     vis_save_path = os.path.join(all_masks_vis_folder, vis_filename)
            
        #     # Lưu ảnh
        #     cv2.imwrite(vis_save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        # print(f"All mask visualizations saved to: {all_masks_vis_folder}")        
        # --- Part 3: Extract the object's point cloud ---
        depth_intrinsics = {'width': W, 'height': H, 'fx': depth_fx, 'fy': depth_fy, 'cx': depth_cx, 'cy': depth_cy}
        object_pcd = extract_point_cloud_from_mask(depth_image, rgb_image, segmentation_mask, depth_intrinsics)
        o3d.io.write_point_cloud(output_ply_path, object_pcd)
        print(f"Saved segmented object point cloud to {output_ply_path}")
        # object_mesh = create_mesh_from_pointcloud(object_pcd)
        # if object_mesh is not None and object_mesh.has_triangles():
        #     o3d.io.write_triangle_mesh(output_mesh_path, object_mesh)
        #     print(f"Saved reconstructed 3D mesh to {output_mesh_path}")
        # else:
        #     print("Skipping mesh saving as reconstruction failed or produced an empty mesh.")
        # --- Tìm vector pháp tuyến ---
        final_normal = find_surface_normal(depth_image, segmentation_mask, depth_intrinsics, tilt_threshold_deg=7.0)
        # final_normal = find_normal_from_obb(object_pcd)
        if final_normal is None:
            print("Error: Could not determine surface normal.")
            return None, None

        # --- Part 4 & 5: Calculate Final 3D Pose ---
        oriented_rect = get_oriented_bbox_2d(segmentation_mask)
        if oriented_rect is None:
            print("Could not determine object pose.")
            return None, None

        final_depth_mm = None
        center_x_2d, center_y_2d = oriented_rect[0]
        center_x_int, center_y_int = np.intp(oriented_rect[0])

        # --- Primary Method: Check depth at the geometric 2D center ---
        if (0 <= center_y_int < H and 0 <= center_x_int < W):
            depth_at_center_mm = depth_image[center_y_int, center_x_int]
            if depth_at_center_mm > MIN_DEPTH_THRESHOLD:
                print(f"Success: Using valid depth from 2D center: {depth_at_center_mm} mm")
                final_depth_mm = depth_at_center_mm
            else:
                print(f"Warning: Depth at 2D center ({depth_at_center_mm}) is invalid. Triggering fallback.")
        else:
            print(f"Warning: 2D center ({center_x_int}, {center_y_int}) is outside image bounds. Triggering fallback.")

        # --- Fallback Method: If primary method failed, use robust median ---
        if final_depth_mm is None:
            final_depth_mm = get_robust_depth_from_mask(depth_image, segmentation_mask)

        # --- Final Check: If both methods failed, we cannot proceed ---
        if final_depth_mm is None:
            print("Error: Both primary and fallback depth methods failed. Cannot determine object pose.")
            return None, None

        # --- Proceed with the successfully determined depth ---
        Z_depth = final_depth_mm / 1000.0
        X_depth = (center_x_2d - depth_cx) * Z_depth / depth_fx
        Y_depth = (center_y_2d - depth_cy) * Z_depth / depth_fy
        Y_depth *= 0.95
        X_depth *= 1.02
        # Y_depth *= 0.8
        pose_3d_depth_frame = np.array([X_depth, Y_depth, Z_depth])

        print(f"\nFinal 3D Pose: X={pose_3d_depth_frame[0]:.4f}m, Y={pose_3d_depth_frame[1]:.4f}m, Z={pose_3d_depth_frame[2]:.4f}m")
        print(f"Final Surface Normal: {final_normal}")
              
        # --- Part 7: Visualize and save results ---
        vis_image = visualize_results(
            rgb_image, 
            segmentation_mask, 
            point_prompt, 
            oriented_rect, 
            pose_3d_depth_frame, # Pass the 3D pose
            final_normal,        # Pass the normal vector
            depth_intrinsics     # Pass the camera intrinsics
        )
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_vis_path, vis_image_bgr)
        print(f"Saved visualization to {output_vis_path}")

        # Trả về các giá trị cần thiết để ghi ra CSV
        return pose_3d_depth_frame, final_normal

    except Exception as e:
        print(f"An unexpected error occurred while processing {os.path.basename(rgb_path)}: {e}")
        return None, None

# =================================================================================
# ### THAY ĐỔI ###: HÀM MAIN MỚI ĐỂ XỬ LÝ THƯ MỤC
# =================================================================================
def main():
    """
    Hàm main mới: duyệt qua thư mục, xử lý từng cặp ảnh và ghi kết quả vào file CSV.
    """
    # 1. Tạo thư mục output nếu chưa tồn tại
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Output will be saved to: {OUTPUT_FOLDER}")

    # 2. Khởi tạo model SAM một lần duy nhất để tiết kiệm thời gian
    print("Loading SAM 2.1 model (this may take a moment)...")
    sam = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=DEVICE)
    sam_predictor = SAM2ImagePredictor(sam)
    print("SAM model loaded.")

    # 3. Mở file CSV để ghi
    with open(CSV_OUTPUT_PATH, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # 3.1. Ghi header cho file CSV
        header = ['image_filename', 'x', 'y', 'z', 'Rx', 'Ry', 'Rz']
        csv_writer.writerow(header)
        print(f"CSV file created at: {CSV_OUTPUT_PATH}")

        # 4. Lấy danh sách các file ảnh RGB và sắp xếp để đảm bảo thứ tự
        rgb_files = sorted([f for f in os.listdir(RGB_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))])

        # 5. Lặp qua từng file ảnh
        for filename in rgb_files:
            print(f"\n{'='*20} PROCESSING: {filename} {'='*20}")
            
            # 5.1. Tạo đường dẫn đầy đủ cho file RGB và Depth
            rgb_path = os.path.join(RGB_FOLDER, filename)
            depth_path = os.path.join(DEPTH_FOLDER, filename) 

            # 5.2. Kiểm tra xem file depth tương ứng có tồn tại không
            if not os.path.exists(depth_path):
                print(f"Warning: Depth file not found for {filename}. Skipping.")
                continue

            # 5.3. Tạo đường dẫn output động cho từng file
            base_name = os.path.splitext(filename)[0]
            output_vis_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_visualization.jpg")
            output_ply_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_object_pcd.ply")
            output_mesh_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_object_mesh.ply") # New path
            all_masks_vis_folder = os.path.join(OUTPUT_FOLDER, base_name)
            os.makedirs(all_masks_vis_folder, exist_ok=True)
            
            # ### MODIFIED ###: Pass the new mesh path to the processing function
            center_coords, normal_vector = process_image_pair(
                rgb_path, depth_path, sam_predictor, 
                output_vis_path, output_ply_path, None, all_masks_vis_folder
            )

            # 5.5. Nếu xử lý thành công, ghi kết quả vào file CSV
            # =========================================================
            # ### CHỈ THAY ĐỔI TẠI ĐÂY ###
            # =========================================================
            if center_coords is not None and normal_vector is not None:
                x, y, z = center_coords
                rx, ry, rz = normal_vector
                
                # Tạo một hàng dữ liệu và GHI VÀO FILE SAU KHI LÀM TRÒN
                row_data = [
                    "image_" + filename, 
                    x, 
                    y, 
                    z, 
                    rx, 
                    ry, 
                    rz
                ]
                csv_writer.writerow(row_data)
                print(f"Successfully processed and saved data for {filename}")
            else:
                print(f"Failed to process {filename}. Skipping CSV entry.")

    print(f"\n\nPipeline finished. All results saved in {CSV_OUTPUT_PATH}")
    # ### THAY ĐỔI ###: Thêm logic để tạo file zip
    print("\n--- Creating submission zip file ---")
    try:
        zip_path = os.path.join(OUTPUT_FOLDER, 'task3.zip')
        
        # Get the absolute path of the currently running script
        # __file__ gives the path to the current script
        current_script_path = os.path.abspath(__file__)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add the CSV file to the zip
            if os.path.exists(CSV_OUTPUT_PATH):
                # os.path.basename ensures we don't save the full directory structure
                zf.write(CSV_OUTPUT_PATH, arcname=os.path.basename(CSV_OUTPUT_PATH))
                print(f"Added '{os.path.basename(CSV_OUTPUT_PATH)}' to zip.")
            else:
                print(f"Warning: CSV file '{CSV_OUTPUT_PATH}' not found. It will be missing from the zip.")

            # Add the current script to the zip with the required name 'task3.py'
            if os.path.exists(current_script_path):
                zf.write(current_script_path, arcname='task3.py')
                print(f"Added current script as 'task3.py' to zip.")
            else:
                print(f"Warning: Script file '{current_script_path}' not found. It will be missing from the zip.")

        print(f"\nSuccessfully created submission file: {zip_path}")

    except Exception as e:
        print(f"\nAn error occurred while creating the zip file: {e}")

if __name__ == '__main__':
    main()