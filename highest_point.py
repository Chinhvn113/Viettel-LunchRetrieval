import cv2
import numpy as np
from scipy import ndimage

# ==================== CAMERA PARAMETERS ====================
# Depth intrinsics
depth_fx = 650.0616455078125
depth_fy = 650.0616455078125
depth_cx = 649.5928055078125
depth_cy = 360.9415588378906

# Color intrinsics (nếu cần transform sang color frame)
color_fx = 643.90087890625
color_fy = 643.1365356445312
color_cx = 650.2113037109375
color_cy = 355.79550326171875

# ROI parameters
ROI_X, ROI_Y, ROI_W, ROI_H = 560, 150, 300, 330

# ==================== ADVANCED PARAMETERS ====================
EDGE_MARGIN = 10  # Loại bỏ N pixels từ mép ROI
MIN_DEPTH_THRESHOLD = 300  # mm - loại bỏ depth quá nhỏ (noise)
MAX_DEPTH_THRESHOLD = 5000  # mm - loại bỏ depth quá lớn
MEDIAN_FILTER_SIZE = 3  # Kích thước kernel median filter (giảm noise)
CLUSTER_SIZE = 5  # Tìm vùng NxN pixels có depth trung bình nhỏ nhất

# ==================== LOAD IMAGES ====================
def load_images(depth_path, rgb_path):
    """Load depth and RGB images"""
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    
    if depth_image is None or rgb_image is None:
        raise ValueError("Cannot load images!")
    
    return depth_image, rgb_image

# ==================== FIND CLOSEST POINT (ROBUST VERSION) ====================
def find_closest_point_in_roi(depth_image, roi_x, roi_y, roi_w, roi_h, 
                               edge_margin=EDGE_MARGIN,
                               min_depth=MIN_DEPTH_THRESHOLD,
                               max_depth=MAX_DEPTH_THRESHOLD,
                               median_kernel=MEDIAN_FILTER_SIZE,
                               cluster_size=CLUSTER_SIZE,
                               verbose=True):
    """
    Tìm điểm có depth nhỏ nhất (gần camera nhất) trong ROI một cách robust
    
    Parameters:
        edge_margin: Số pixel loại bỏ từ mép ROI
        min_depth: Depth tối thiểu hợp lệ (mm)
        max_depth: Depth tối đa hợp lệ (mm)
        median_kernel: Kích thước kernel cho median filter
        cluster_size: Tìm vùng NxN có depth trung bình nhỏ nhất
        verbose: In thông tin debug
    
    Returns:
        pixel_x, pixel_y: Tọa độ pixel trong toàn bộ ảnh
        depth_value: Giá trị depth (mm)
        point_3d: Tọa độ 3D (X, Y, Z) trong hệ camera depth (m)
    """
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

# ==================== VISUALIZATION ====================
def visualize_result(rgb_image, pixel_x, pixel_y, depth_value, point_3d, 
                    roi_x, roi_y, roi_w, roi_h, edge_margin=EDGE_MARGIN):
    """
    Vẽ kết quả lên ảnh RGB với nhiều thông tin hơn
    """
    result_image = rgb_image.copy()
    
    # 1. Vẽ ROI ngoài (bao gồm edge)
    cv2.rectangle(result_image, 
                  (roi_x, roi_y), 
                  (roi_x + roi_w, roi_y + roi_h), 
                  (0, 255, 0), 2)
    
    # 2. Vẽ ROI trong (loại trừ edge)
    if edge_margin > 0:
        cv2.rectangle(result_image, 
                      (roi_x + edge_margin, roi_y + edge_margin), 
                      (roi_x + roi_w - edge_margin, roi_y + roi_h - edge_margin), 
                      (255, 255, 0), 1)
        cv2.putText(result_image, "ROI (outer)", 
                    (roi_x, roi_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(result_image, f"Valid area (-{edge_margin}px edge)", 
                    (roi_x + edge_margin, roi_y + edge_margin - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    else:
        cv2.putText(result_image, "ROI", 
                    (roi_x, roi_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 3. Vẽ điểm gần nhất
    # Crosshair lớn
    line_length = 25
    cv2.line(result_image, 
             (pixel_x - line_length, pixel_y), 
             (pixel_x + line_length, pixel_y), 
             (0, 0, 255), 3)
    cv2.line(result_image, 
             (pixel_x, pixel_y - line_length), 
             (pixel_x, pixel_y + line_length), 
             (0, 0, 255), 3)
    
    # Circle
    cv2.circle(result_image, (pixel_x, pixel_y), 15, (0, 0, 255), 3)
    cv2.circle(result_image, (pixel_x, pixel_y), 3, (255, 255, 255), -1)
    
    # 4. Hiển thị thông tin
    X, Y, Z = point_3d
    info_texts = [
        "CLOSEST POINT",
        f"Pixel: ({pixel_x}, {pixel_y})",
        f"Depth: {depth_value}mm ({Z:.3f}m)",
        f"3D: ({X:.3f}, {Y:.3f}, {Z:.3f})m"
    ]
    
    # Vẽ background cho text
    text_x = pixel_x + 25
    text_y_start = pixel_y - 60
    
    # Adjust text position if too close to edge
    if text_y_start < 20:
        text_y_start = pixel_y + 30
    if text_x + 250 > result_image.shape[1]:
        text_x = pixel_x - 280
    
    for i, text in enumerate(info_texts):
        y_pos = text_y_start + i * 22
        # Background
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(result_image, 
                      (text_x - 5, y_pos - text_h - 3), 
                      (text_x + text_w + 5, y_pos + baseline + 3), 
                      (0, 0, 0), -1)
        # Text
        color = (0, 255, 255) if i == 0 else (255, 255, 255)
        thickness = 2 if i == 0 else 1
        cv2.putText(result_image, text, 
                    (text_x, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
    return result_image

# ==================== MAIN FUNCTION ====================
def main(depth_path, rgb_path, output_path='result.jpg', 
         edge_margin=EDGE_MARGIN, 
         median_kernel=MEDIAN_FILTER_SIZE,
         cluster_size=CLUSTER_SIZE,
         verbose=True):
    """
    Main pipeline với robust filtering
    """
    if verbose:
        print("="*60)
        print("ROBUST CLOSEST POINT DETECTION")
        print("="*60)
        print(f"Configuration:")
        print(f"  Edge margin: {edge_margin}px")
        print(f"  Median filter: {median_kernel}x{median_kernel}")
        print(f"  Cluster averaging: {cluster_size}x{cluster_size}")
        print(f"  Depth range: {MIN_DEPTH_THRESHOLD}-{MAX_DEPTH_THRESHOLD}mm")
    
    print("\nLoading images...")
    depth_image, rgb_image = load_images(depth_path, rgb_path)
    if verbose:
        print(f"Depth image: {depth_image.shape}, {depth_image.dtype}")
        print(f"RGB image: {rgb_image.shape}")
    
    pixel_x, pixel_y, depth_value, point_3d = find_closest_point_in_roi(
        depth_image, ROI_X, ROI_Y, ROI_W, ROI_H,
        edge_margin=edge_margin,
        median_kernel=median_kernel,
        cluster_size=cluster_size,
        verbose=verbose
    )
    
    print("\nVisualizing result...")
    result_image = visualize_result(
        rgb_image, pixel_x, pixel_y, depth_value, point_3d,
        ROI_X, ROI_Y, ROI_W, ROI_H, edge_margin
    )
    
    print(f"Saving result to {output_path}...")
    cv2.imwrite(output_path, result_image)
    
    print("\nDisplaying result (press any key to close)...")
    # Resize for display if image is too large
    display_image = result_image.copy()
    h, w = display_image.shape[:2]
    if w > 1920:
        scale = 1920 / w
        display_image = cv2.resize(display_image, None, fx=scale, fy=scale)
    
    cv2.imshow('Robust Closest Point Detection', display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    
    return pixel_x, pixel_y, depth_value, point_3d

# ==================== USAGE ====================
if __name__ == "__main__":
    # Thay đường dẫn ảnh của bạn
    DEPTH_PATH = "/mlcv2/WorkingSpace/Personal/chinhnm/LunchBox/Viettel/0000_depth.png"
    RGB_PATH = "/mlcv2/WorkingSpace/Personal/chinhnm/LunchBox/Viettel/0000_rgb.png"
    OUTPUT_PATH = "result_robust.jpg"
    
    # Chạy với cấu hình mặc định (recommended)
    pixel_x, pixel_y, depth_value, point_3d = main(
        DEPTH_PATH, RGB_PATH, OUTPUT_PATH,
        edge_margin=10,      # Loại bỏ 10px từ mép
        median_kernel=1,     # Median filter 3x3
        cluster_size=1       # Tìm vùng 5x5 có depth TB nhỏ nhất61
    )
    
    # Hoặc với edge margin lớn hơn nếu vẫn có vấn đề:
    # pixel_x, pixel_y, depth_value, point_3d = main(
    #     DEPTH_PATH, RGB_PATH, OUTPUT_PATH,
    #     edge_margin=15,
    #     median_kernel=5,
    #     cluster_size=7
    # )



    