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
import random

# ==================== CONFIGURATION ====================
SEED_SAM = 0
SEED_RANSAC = 44
np.random.seed(SEED_SAM)
random.seed(SEED_SAM)
torch.manual_seed(SEED_SAM)
try:
    o3d.utility.random.seed(SEED_RANSAC)  # Từ Open3D 0.18.0 trở lên
except AttributeError:
    print("⚠️ open3d.utility.random.seed() không hỗ trợ trong version hiện tại.")
# ### THAY ĐỔI ###: Chuyển từ đường dẫn file sang đường dẫn thư mục
# --- Folder Paths ---
RGB_FOLDER = "/mlcv2/WorkingSpace/Personal/chinhnm/LunchBox/Viettel/Data/ThiSinh/rgb"
DEPTH_FOLDER = "/mlcv2/WorkingSpace/Personal/chinhnm/LunchBox/Viettel/Data/ThiSinh/depth"
OUTPUT_FOLDER = "Viettel/visualize_output_private" # Thư mục chứa ảnh và file PLY
CSV_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, "Submission_3D.csv") # Đường dẫn file CSV kết quả

# --- SAM Checkpoint ---
SAM2_CHECKPOINT = "./checkpoints/checkpoint.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAM_IN_SIZE = 1024

# --- Camera Intrinsics (Depth Camera) ---
depth_fx = 650.0616455078125
depth_fy = 650.0616455078125
depth_cx = 649.5928955078125
depth_cy = 360.9415588378906

# Color intrinsics (nếu cần transform sang color frame)
R_depth_to_color = np.array([
    [0.9999898076057434, -0.00020347206736914814, -0.004507721401751041],
    [0.00018898719281423837, 0.9999948143959045, -0.0032135415822267532],
    [0.004508351907134056, 0.003212657058611512, 0.9999846816062927]
])

# Vector tịnh tiến (Translation Vector t) in meters
t_depth_to_color = np.array([-0.05905, 8.67399e-5, 0.00041])

# 2. Color Camera Intrinsics
color_fx = 643.90087890625
color_fy = 643.1365356445312
color_cx = 650.2113037109375
color_cy = 355.79559326171875

# 3. Color Camera Distortion Coefficients
# Model: Brown-Conrady (k1, k2, p1, p2, k3)
color_coeffs = np.array([-0.05658450722694397, 0.06544225662946701, 
                         -0.0008694113348610699, 0.00016751799557823688, 
                         -0.020957745611667633])

# --- Processing Parameters ---
ROI_X, ROI_Y, ROI_W, ROI_H = 560, 150, 300, 330
# ==================== ADVANCED PARAMETERS ====================
EDGE_MARGIN = 10
MIN_DEPTH_THRESHOLD = 300
MAX_DEPTH_THRESHOLD = 5000
MEDIAN_FILTER_SIZE = 1
CLUSTER_SIZE = 1
NORMAL_VIS_LENGTH = 0.1
POISSON_DEPTH = 9



# =================================================================================
# CÁC HÀM PHỤ (giữ nguyên không thay đổi)
# =================================================================================
def scale_points(points, sx, sy):
    """Scale a list of (x,y) points by (sx, sy) with rounding."""
    if points is None: return None
    out = []
    for x, y in points:
        out.append((int(round(x * sx)), int(round(y * sy))))
    return out

def scale_point(p, sx, sy):
    x, y = p
    return (int(round(x * sx)), int(round(y * sy)))

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
def centroid_from_mask_in_box(seg_mask: np.ndarray, rect, shrink=0.90, erode_px=2):
    """
    Returns (cx, cy) centroid of seg_mask constrained inside the rotated box.
    Falls back with a looser box if the first one is empty.
    Also returns the final in-box mask used.
    """
    H, W = seg_mask.shape
    # first, shrink a bit to avoid touching neighbors/edges
    in_box = mask_from_rotated_rect(rect, (H, W), scale=shrink, erode_px=erode_px)
    m = seg_mask & in_box
    if m.sum() == 0:
        # relax constraints if too small/empty
        in_box = mask_from_rotated_rect(rect, (H, W), scale=0.98, erode_px=0)
        m = seg_mask & in_box
        if m.sum() == 0:
            return None, in_box  # still empty

    c = calculate_mask_centroid(m)
    return c, m
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
def core_top_mask_from_seg(depth_mm, seg_mask, rect,
                           shrink=0.90, erode_px=2,
                           min_depth=300, delta_mm=25):
    H, W = depth_mm.shape
    # 1) shrinked rotated box to avoid touching neighbors
    in_box = mask_from_rotated_rect(rect, (H, W), scale=shrink, erode_px=erode_px)
    cand = seg_mask & in_box

    if cand.sum() == 0:
        # relax if empty
        in_box = mask_from_rotated_rect(rect, (H, W), scale=0.96, erode_px=0)
        cand = seg_mask & in_box

    if cand.sum() == 0:
        return cand  # empty, caller should handle

    # 2) robust depth gate around the *mask* depth, not a single pixel
    vals = depth_mm[cand]
    vals = vals[vals > min_depth]
    if len(vals) == 0:
        return np.zeros_like(seg_mask, dtype=bool)

    # Use median (or np.quantile / mode estimate) as the object top depth
    z0 = float(np.median(vals))
    gate = (depth_mm >= (z0 - delta_mm)) & (depth_mm <= (z0 + delta_mm))

    core = cand & gate

    # 3) light erosion to remove thin rims/edges
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px*2+1, erode_px*2+1))
        core = cv2.erode((core.astype(np.uint8)*255), k, iterations=1) > 0

    return core

def find_closest_point_in_roi(
    depth_image, roi_x, roi_y, roi_w, roi_h,
    edge_margin=EDGE_MARGIN,
    min_depth=MIN_DEPTH_THRESHOLD,
    max_depth=MAX_DEPTH_THRESHOLD,
    median_kernel=MEDIAN_FILTER_SIZE,
    cluster_size=CLUSTER_SIZE,
    tie_mm=5,
    verbose=True
):
    # 1) Crop ROI
    roi_depth = depth_image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()

    if median_kernel > 1:
        roi_depth_filtered = cv2.medianBlur(roi_depth, median_kernel)
    else:
        roi_depth_filtered = roi_depth.copy()

    # 2) Valid mask + trim edges
    valid_mask = (roi_depth_filtered > min_depth) & (roi_depth_filtered < max_depth)
    if edge_margin > 0:
        edge = np.ones_like(valid_mask, dtype=bool)
        edge[:edge_margin, :] = False
        edge[-edge_margin:, :] = False
        edge[:, :edge_margin] = False
        edge[:, -edge_margin:] = False
        valid_mask &= edge

    if not np.any(valid_mask):
        raise ValueError("No valid depth values in ROI after filtering!")

    if cluster_size > 1:
        # 3A) Local mean depth (cluster window)
        roi_f = roi_depth_filtered.astype(float)
        roi_f[~valid_mask] = np.nan

        # uniform_filter doesn't handle NaN; build masked average
        # sum and count with constant padding = 0
        k = cluster_size
        sum_img = ndimage.uniform_filter(np.nan_to_num(roi_f, nan=0.0), size=k, mode='constant', cval=0.0) * (k*k)
        cnt_img = ndimage.uniform_filter(valid_mask.astype(np.float32), size=k, mode='constant', cval=0.0) * (k*k)
        with np.errstate(invalid='ignore', divide='ignore'):
            local_mean = sum_img / np.maximum(cnt_img, 1e-9)
            local_mean[ cnt_img < 1 ] = np.inf  # invalid windows

        # min local mean among valid windows
        valid_local = (cnt_img >= 1) & valid_mask
        min_local = np.min(local_mean[valid_local])

        # 4A) Tie set = within <= tie_mm of min local mean
        candidates = valid_local & (local_mean <= (min_local + tie_mm))

        if not np.any(candidates):
            # fallback to strict min
            min_idx = np.argmin(np.where(valid_local, local_mean, np.inf))
            ly, lx = np.unravel_index(min_idx, roi_depth.shape)
        else:
            # prefer top: smallest y, then smallest x
            ys, xs = np.where(candidates)
            order = np.lexsort((xs, ys))  # primary: ys, secondary: xs
            ly, lx = ys[order[0]], xs[order[0]]
    else:
        # 3B) Single-pixel min
        roi_masked = roi_depth_filtered.copy().astype(float)
        roi_masked[~valid_mask] = np.inf
        min_val = np.min(roi_masked)

        # 4B) Tie set = within <= tie_mm of global minimum
        candidates = valid_mask & (roi_depth_filtered <= (min_val + tie_mm))
        if not np.any(candidates):
            min_idx = np.argmin(roi_masked)
            ly, lx = np.unravel_index(min_idx, roi_depth.shape)
        else:
            ys, xs = np.where(candidates)
            order = np.lexsort((xs, ys))  # top-most, then left-most
            ly, lx = ys[order[0]], xs[order[0]]

    # 5) Back to full-image coords
    pixel_x = roi_x + lx
    pixel_y = roi_y + ly
    depth_value = float(depth_image[pixel_y, pixel_x])

    if verbose:
        print(f"\n=== RESULT ===")
        print(f"Closest (with tie <= {tie_mm}mm, pref top): ({pixel_x}, {pixel_y}), depth={depth_value:.1f} mm")

    # 6) Unproject to 3D (depth intrinsics)
    Z = depth_value / 1000.0
    X = (pixel_x - depth_cx) * Z / depth_fx
    Y = (pixel_y - depth_cy) * Z / depth_fy
    point_3d = (X, Y, Z)

    if verbose:
        print(f"3D coordinates: X={X:.4f}m, Y={Y:.4f}m, Z={Z:.4f}m")

    return pixel_x, pixel_y, depth_value, point_3d


def find_farthest_point_in_roi(depth_image, roi_x, roi_y, roi_w, roi_h,
                               edge_margin=EDGE_MARGIN,
                               min_depth=MIN_DEPTH_THRESHOLD,
                               max_depth=MAX_DEPTH_THRESHOLD,
                               median_kernel=MEDIAN_FILTER_SIZE):
    """
    Finds the point with the maximum depth value within the ROI, intended to be a
    background point for a negative SAM prompt.
    Returns (pixel_x, pixel_y) or raises ValueError if no valid point is found.
    """
    # 1. Crop ROI from depth image
    roi_depth = depth_image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()

    # 2. Apply median filter to reduce noise
    if median_kernel > 1:
        roi_depth_filtered = cv2.medianBlur(roi_depth, median_kernel)
    else:
        roi_depth_filtered = roi_depth.copy()

    # 3. Create a valid mask within the depth range
    valid_mask = (roi_depth_filtered > min_depth) & (roi_depth_filtered < max_depth)

    # 4. Exclude edge pixels from consideration
    if edge_margin > 0:
        edge_mask = np.ones_like(valid_mask, dtype=bool)
        edge_mask[:edge_margin, :] = False
        edge_mask[-edge_margin:, :] = False
        edge_mask[:, :edge_margin] = False
        edge_mask[:, -edge_margin:] = False
        valid_mask = valid_mask & edge_mask

    if not np.any(valid_mask):
        raise ValueError("No valid background depth values in ROI to select a negative prompt!")

    # 5. Find the pixel with the LARGEST depth value
    # Temporarily set invalid pixels to 0 so they are ignored by argmax
    roi_depth_masked = roi_depth_filtered.copy()
    roi_depth_masked[~valid_mask] = 0

    # Find the index of the maximum depth value
    max_idx = np.argmax(roi_depth_masked)
    local_y, local_x = np.unravel_index(max_idx, roi_depth.shape)

    # 6. Convert local ROI coordinates to global image coordinates
    pixel_x = roi_x + local_x
    pixel_y = roi_y + local_y

    print(f"Found farthest point (negative prompt) at: ({pixel_x}, {pixel_y})")
    
    return pixel_x, pixel_y
def mask_from_rotated_rect(rect, shape_hw, scale=0.90, erode_px=0):
    """
    Build a boolean mask of a rotated rectangle (optionally shrunk).
    rect: cv2.minAreaRect output -> ((cx,cy), (w,h), angle)
    shape_hw: (H, W) of the target image
    scale: <1 shrinks the box inward; >1 expands
    erode_px: extra erosion in pixels to stay well inside the box
    """
    H, W = shape_hw
    (cx, cy), (w, h), ang = rect
    w2, h2 = max(w * scale, 1.0), max(h * scale, 1.0)
    rect_scaled = ((cx, cy), (w2, h2), ang)
    box = cv2.boxPoints(rect_scaled)
    box = np.int32(np.round(box))

    m = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(m, box, 255)

    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px*2+1, erode_px*2+1))
        m = cv2.erode(m, k, iterations=1)

    return m.astype(bool)
# === NEW: sample K negative points from the conveyor belt (depth background) ===
def sample_belt_negatives(depth_image, roi_x, roi_y, roi_w, roi_h, k=2,
                          edge_margin=EDGE_MARGIN,
                          min_depth=MIN_DEPTH_THRESHOLD,
                          max_depth=MAX_DEPTH_THRESHOLD,
                          quantile=0.90,      # take top 10% farthest depth as belt
                          min_pair_dist=25):  # px, keep the two points apart
    roi = depth_image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
    valid = (roi > min_depth) & (roi < max_depth)

    # exclude margins to avoid rails/walls
    if edge_margin > 0:
        inner = np.zeros_like(valid, dtype=bool)
        inner[edge_margin:-edge_margin, edge_margin:-edge_margin] = True
        valid &= inner

    if not np.any(valid):
        # fallback: center-left/right inside ROI
        return [(roi_x + roi_w//4, roi_y + roi_h//2),
                (roi_x + 3*roi_w//4, roi_y + roi_h//2)]

    # depth threshold for "belt" (farthest region)
    dvals = roi[valid]
    thr = np.quantile(dvals, quantile)
    cand_mask = valid & (roi >= thr)
    ys, xs = np.where(cand_mask)
    if len(xs) == 0:
        # fallback to absolute farthest pixels
        max_idx = np.argmax(roi * valid)
        y0, x0 = np.unravel_index(max_idx, roi.shape)
        # zero out a disk around the first to get a second
        m = valid.copy().astype(np.uint8)*255
        cv2.circle(m, (x0, y0), 25, 0, -1)
        roi2 = roi.copy(); roi2[m == 0] = 0
        if np.count_nonzero(roi2) == 0:
            return [(roi_x + x0, roi_y + y0),
                    (roi_x + min(x0+40, roi_w-1), roi_y + y0)]
        max2 = np.argmax(roi2)
        y1, x1 = np.unravel_index(max2, roi2.shape)
        return [(roi_x + x0, roi_y + y0), (roi_x + x1, roi_y + y1)]

    # random sample while enforcing pair distance
    idx = np.random.permutation(len(xs))
    picked = []
    for i in idx:
        gx, gy = roi_x + xs[i], roi_y + ys[i]
        ok = True
        for (px, py) in picked:
            if (gx-px)**2 + (gy-py)**2 < min_pair_dist**2:
                ok = False; break
        if ok:
            picked.append((gx, gy))
            if len(picked) >= k:
                break

    # ensure we always return k points (fallback to corners)
    while len(picked) < k:
        fallback = (roi_x + np.random.randint(edge_margin, roi_w-edge_margin),
                    roi_y + np.random.randint(edge_margin, roi_h-edge_margin))
        picked.append(fallback)

    return picked
def sample_adjacent_object_negatives(depth_mm, seed_xy, r_pixels=80, k=2):
    """Pick negatives from the largest blob *outside* a disk around the seed."""
    y, x = int(seed_xy[1]), int(seed_xy[0])
    h, w = depth_mm.shape
    # candidate = valid depth
    valid = (depth_mm > MIN_DEPTH_THRESHOLD).astype(np.uint8)
    # mask out a disk around seed so we avoid the target object region
    disk = np.zeros_like(valid); cv2.circle(disk, (x,y), r_pixels, 1, -1)
    cand = valid & (1 - disk)

    num, labels = cv2.connectedComponents(cand, connectivity=8)[0:2]
    if num <= 1:  # no components
        return []

    # find largest component
    areas = [(labels==i).sum() for i in range(1, num)]
    i_max = np.argmax(areas) + 1
    ys, xs = np.where(labels == i_max)
    if len(xs) == 0: return []

    idx = np.random.choice(len(xs), size=min(k, len(xs)), replace=False)
    return [(int(xs[i]), int(ys[i])) for i in idx]

def keep_seed_component(mask: np.ndarray, seed_xy, open_ksize=3):
    """
    Keep only the connected component (8-conn) that contains the seed (x,y).
    Optionally break thin bridges with a small opening first.
    """
    m = (mask.astype(np.uint8) * 255)
    if open_ksize and open_ksize > 0:
        k = np.ones((open_ksize, open_ksize), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)

    h, w = m.shape
    x, y = int(seed_xy[0]), int(seed_xy[1])
    x = np.clip(x, 0, w - 1); y = np.clip(y, 0, h - 1)

    # if seed is not inside mask, expand slightly and try again
    if m[y, x] == 0:
        k = np.ones((3, 3), np.uint8)
        m = cv2.dilate(m, k, iterations=1)
        if m[y, x] == 0:
            # nothing we can do; return original mask
            return mask.astype(bool)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    lbl = labels[y, x]
    return (labels == lbl)
def depth_gate_around_seed(depth_mm, seed_xy, delta_mm=35, min_depth=300):
    x, y = int(seed_xy[0]), int(seed_xy[1])
    h, w = depth_mm.shape
    x = np.clip(x, 0, w-1); y = np.clip(y, 0, h-1)
    z0 = int(depth_mm[y, x])

    # If center depth is invalid, use local median
    if z0 < min_depth:
        r = 5
        patch = depth_mm[max(0,y-r):y+r+1, max(0,x-r):x+r+1]
        vals = patch[patch > min_depth]
        if vals.size == 0:
            return np.zeros_like(depth_mm, dtype=bool)
        z0 = int(np.median(vals))

    return (depth_mm >= (z0 - delta_mm)) & (depth_mm <= (z0 + delta_mm))
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


def segment_object_with_sam_old(predictor, rgb_image, point_prompt, roi_w, roi_h, neg_points=None):
    """
    Segments an object with SAM. Supports optional negative prompts from the belt.
    """
    print("\n--- Part 2: Segmenting object with SAM (pos + optional negatives) ---")
    predictor.set_image(rgb_image)

    pos = np.array(point_prompt, dtype=np.int32)[None, :]         # shape (1,2)
    if neg_points is not None and len(neg_points) > 0:
        neg = np.array(neg_points, dtype=np.int32)                # shape (K,2)
        pts = np.vstack([pos, neg])
        labels = np.array([1] + [0]*len(neg), dtype=np.int32)
    else:
        pts = pos
        labels = np.array([1], dtype=np.int32)

    mask = predictor.predict(
        point_coords=pts,
        point_labels=labels,
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
            depth_image, ROI_X, ROI_Y, ROI_W, ROI_H, verbose=False
        )
        point_prompt = np.array([pixel_x, pixel_y])
        sx = SAM_IN_SIZE / float(W)
        sy = SAM_IN_SIZE / float(H)
        rgb_for_sam = cv2.resize(rgb_image, (SAM_IN_SIZE, SAM_IN_SIZE), interpolation=cv2.INTER_LINEAR)

        # scale prompts to the resized image
        pt_sam = scale_point((int(point_prompt[0]), int(point_prompt[1])), sx, sy)
        # === NEW: pick 2 belt negatives in ROI using depth ===
        neg_points = sample_belt_negatives(depth_image, ROI_X, ROI_Y, ROI_W, ROI_H, k=2)
        neg_points_sam = [scale_point(p, sx, sy) for p in neg_points]
        print(f"Negative prompts (belt): {neg_points}")

        # --- Part 2: Segment the object using SAM 2.1 with negatives ---
        # segmentation_mask = segment_object_with_sam_old(
        #     sam_predictor, rgb_image, point_prompt, ROI_W, ROI_H, neg_points=neg_points
        # )
        segmentation_mask_1024 = segment_object_with_sam_old(
            sam_predictor,
            rgb_for_sam,
            np.array(pt_sam),
            ROI_W, ROI_H,
            neg_points=neg_points_sam  # uncomment if using negatives
        )
        segmentation_mask = cv2.resize(
            (segmentation_mask_1024.astype(np.uint8) * 255),
            (W, H),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)              
        # --- Part 3: Extract the object's point cloud ---
        depth_intrinsics = {'width': W, 'height': H, 'fx': depth_fx, 'fy': depth_fy, 'cx': depth_cx, 'cy': depth_cy}
        object_pcd = extract_point_cloud_from_mask(depth_image, rgb_image, segmentation_mask, depth_intrinsics)
        o3d.io.write_point_cloud(output_ply_path, object_pcd)
        print(f"Saved segmented object point cloud to {output_ply_path}")
        # --- Tìm vector pháp tuyến ---
        print("Cleaning point cloud with Statistical Outlier Removal...")
        cleaned_pcd, ind = object_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"Removed {len(object_pcd.points) - len(cleaned_pcd.points)} outlier points.")
        final_normal = find_surface_normal(depth_image, segmentation_mask, depth_intrinsics, tilt_threshold_deg=7.0, ransac_distance_threshold=0.01)
        if final_normal is None:
            print("Error: Could not determine surface normal.")
            return None, None
        mask_for_centroid = segmentation_mask.copy()
        if np.array_equal(final_normal, np.array([0,0,1])):
                gate = depth_gate_around_seed(depth_image, point_prompt, delta_mm=35)
                mask_gated = segmentation_mask & gate
                mask_seed = keep_seed_component(mask_gated, point_prompt, open_ksize=3)
                if mask_seed.sum() < 200:  # fallback if too small
                    mask_seed = keep_seed_component(segmentation_mask, point_prompt, open_ksize=0)
                mask_for_centroid = mask_seed
        # --- Part 4 & 5: Calculate Final 3D Pose ---
        oriented_rect = get_oriented_bbox_2d(mask_for_centroid)
        final_depth_mm = None
        center_x_2d, center_y_2d = oriented_rect[0]
        center_x_int, center_y_int = np.intp(oriented_rect[0])
        if oriented_rect is None:
            print("Could not determine object pose.")
            return None, None
        normal_core = core_top_mask_from_seg(
            depth_image, segmentation_mask, oriented_rect,
            shrink=0.90, erode_px=2, delta_mm=25
        )

        if normal_core.sum() < 300:
            normal_core = core_top_mask_from_seg(
                depth_image, segmentation_mask, oriented_rect,
                shrink=0.96, erode_px=0, delta_mm=30
            )
        final_normal = find_surface_normal(
            depth_image, normal_core, depth_intrinsics,
            tilt_threshold_deg=7.0, ransac_distance_threshold=0.01
        )

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
        # Y_depth *= 0.92
        # X_depth *= 1.02
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