import numpy as np
import open3d as o3d
from typing import Dict, Optional
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
try:
    o3d.utility.random.seed(SEED)  # Từ Open3D 0.18.0 trở lên
except AttributeError:
    print("⚠️ open3d.utility.random.seed() không hỗ trợ trong version hiện tại.")

def find_surface_normal(
    depth_image: np.ndarray, 
    mask: np.ndarray, 
    depth_intrinsics: Dict, 
    ransac_distance_threshold: float = 0.01,
    min_plane_points: int = 200,
    tilt_threshold_deg: float = 8.0
) -> Optional[np.ndarray]:
    print("\n=== Finding Robust Surface Normal Vector ===")
    
    # --- 1. Tạo Point Cloud (Tương tự code cũ) ---
    pixels_y, pixels_x = np.where(mask)
    if len(pixels_x) < min_plane_points:
        print(f"Warning: Not enough points ({len(pixels_x)}). Returning None.")
        return None
        
    Z = depth_image[pixels_y, pixels_x] / 1000.0
    X = (pixels_x - depth_intrinsics['cx']) * Z / depth_intrinsics['fx']
    Y = (pixels_y - depth_intrinsics['cy']) * Z / depth_intrinsics['fy']
    points_3d = np.stack((X, Y, Z), axis=-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    # --- 2. Tìm mặt phẳng bằng RANSAC (Tương tự code cũ) ---
    try:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=ransac_distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
    except Exception as e:
        print(f"Error during plane segmentation: {e}. Returning None.")
        return None

    if not inliers:
        print("Warning: RANSAC could not find a plane. Returning None.")
        return None

    # --- 3. Chuẩn hóa và Định hướng Vector Pháp tuyến ---
    # Vector pháp tuyến ban đầu từ RANSAC
    normal_vector = np.array(plane_model[:3])
    
    # Chuẩn hóa độ dài vector thành 1
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    
    # **BƯỚC QUAN TRỌNG: Đảm bảo vector luôn hướng ra xa camera**
    # Giả sử camera nhìn xuống, trục Z dương hướng xuống. Chúng ta muốn thành phần z
    # của vector pháp tuyến phải dương.
    if normal_vector[2] < 0:
        normal_vector = -normal_vector # Đảo chiều vector
        
    print(f"Found and oriented normal vector: {normal_vector}")

    # --- 4. Áp dụng ngưỡng nghiêng (Tilt Thresholding) ---
    # Vector lý tưởng của mặt phẳng không nghiêng (song song mặt phẳng ảnh)
    flat_normal = np.array([0, 0, 1])
    
    # Tính góc giữa vector tìm được và vector lý tưởng
    dot_product = np.clip(np.dot(normal_vector, flat_normal), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    print(f"Calculated tilt angle: {angle_deg:.2f} degrees")

    # --- 5. Trả về kết quả cuối cùng ---
    if angle_deg < tilt_threshold_deg:
        print(f"Tilt is less than {tilt_threshold_deg} degrees. Assuming flat surface.")
        return flat_normal
    else:
        print("Tilt is significant. Using calculated normal vector.")
        return normal_vector
