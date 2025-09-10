import open3d as o3d
import numpy as np
import torch
import turboreg_gpu
from scipy.linalg import null_space

def to_homogeneous(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])

def apply_homography(H, X, debug=False):
    X_h = to_homogeneous(X)
    X_trans = (H @ X_h.T).T
    if debug:
        print(X_trans[:, 3])
    return X_trans[:, :3] / X_trans[:, 3:]

def apply_homography_batch(H_batch: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Efficiently apply batched 4x4 homographies to 3D points.
    
    Args:
        H_batch: Tensor of shape (B, 4, 4)
        X:       Tensor of shape (N, 3)
    Returns:
        Transformed points: Tensor of shape (B, N, 3)
    """
    B = H_batch.shape[0]
    N = X.shape[0]
    
    # Append 1 to each point: (N, 4)
    ones = torch.ones((N, 1), dtype=X.dtype, device=X.device)
    X_h = torch.cat([X, ones], dim=1)  # (N, 4)

    # Apply homographies: (B, 4, 4) x (N, 4)^T → (B, 4, N)
    X_h = X_h.T.unsqueeze(0).expand(B, 4, N)  # (B, 4, N)
    X_trans = torch.bmm(H_batch, X_h)  # (B, 4, N)

    # Perspective divide
    X_trans = X_trans[:, :3, :] / X_trans[:, 3:4, :]  # (B, 3, N)
    
    # Transpose to (B, N, 3)
    return X_trans.permute(0, 2, 1)

def estimate_3D_homography(X_src_batch, X_dst_batch):
    """
    Estimate batch of 3D Homography.
    
    Inputs:
        X_src_batch: (B, N, 3)
        X_dst_batch: (B, N, 3)
        
    Returns:
        H_batch: (B, 4, 4)
    """
    B, N, _ = X_src_batch.shape
    ones = np.ones((B, N))

    x, y, z = X_src_batch[:, :, 0], X_src_batch[:, :, 1], X_src_batch[:, :, 2]
    xp, yp, zp = X_dst_batch[:, :, 0], X_dst_batch[:, :, 1], X_dst_batch[:, :, 2]

    # Prepare matrices
    A = np.zeros((B, 3 * N, 16))

    stacked_X = np.stack([x, y, z, ones], axis=2)  # (B, N, 4)

    # Fill in A
    A[:, 0::3, 0:4] = -stacked_X
    A[:, 0::3, 12:16] = np.stack([x * xp, y * xp, z * xp, xp], axis=2)

    A[:, 1::3, 4:8] = -stacked_X
    A[:, 1::3, 12:16] = np.stack([x * yp, y * yp, z * yp, yp], axis=2)

    A[:, 2::3, 8:12] = -stacked_X
    A[:, 2::3, 12:16] = np.stack([x * zp, y * zp, z * zp, zp], axis=2)

    # Solve using null space
    H_batch = np.zeros((B, 4, 4))
    for i in range(B):
        nullvec = null_space(A[i])
        if nullvec.shape[1] == 0:
            H_batch[i] = np.eye(4)
            continue

        H = nullvec[:, 0].reshape(4, 4)
        if H[3, 3] == 0:
            H_batch[i] = np.eye(4)
            continue

        H = H / H[3, 3]

        det = np.linalg.det(H)
        if np.isnan(det) or det < 0.0001:
            H_batch[i] = np.eye(4)
        else:
            H_batch[i] = H / det**0.25

    return torch.tensor(H_batch, dtype = torch.float32, device='cuda')

def is_planar(X, threshold=5e-2):
    X_centered = X - X.mean(axis=0)
    _, S, _ = np.linalg.svd(X_centered)
    normal_strength = S[-1] / S[0]
    return normal_strength < threshold

def scale(X):
    centroid = X.mean(axis=0)
    X_centered = X - centroid  # move centroid to origin

    # Compute average distance to the origin after centering
    avg_norm = np.linalg.norm(X_centered, axis=1).mean()

    # Desired average distance is sqrt(3)
    desired_avg_norm = np.sqrt(3)

    # Compute the uniform scaling factor
    scale = desired_avg_norm / avg_norm

    # Construct the 4x4 similarity transform matrix
    T = np.eye(4)
    T[:3, :3] *= scale  # apply scaling
    T[:3, 3] = -scale * centroid  # apply translation

    X_h = np.hstack([X, np.ones((X.shape[0], 1))])  # shape: (N, 4)

    # Step 2: Apply the transform
    X_transformed_h = (T @ X_h.T).T  # shape: (N, 4)

    # Step 3: Convert back to 3D (drop the homogeneous coordinate)
    X_transformed = X_transformed_h[:, :3]

    return T, X_transformed

# threshold = 0.01, max_iter = 300, sample_size = 5
def ransac_projective_improved(X1_np, X2_np, threshold=0.01, max_iter=300, sample_size=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 转换数据到GPU
    X1 = torch.tensor(X1_np, dtype=torch.float32, device=device)
    X2 = torch.tensor(X2_np, dtype=torch.float32, device=device)
    N = X1.shape[0]
    
    best_inlier_count = -1
    best_H = None
    
    for i in range(max_iter):
        # 随机采样
        indices = torch.randperm(N, device=device)[:sample_size]
        X1_sample = X1[indices]
        X2_sample = X2[indices]
        
        # 估计单应性矩阵
        try:
            H = estimate_3D_homography(
                X1_sample.cpu().numpy(), 
                X2_sample.cpu().numpy()
            )
            
            # 应用变换并计算误差
            X2_pred = apply_homography(H, X1.cpu().numpy())
            errors = np.linalg.norm(X2_pred - X2.cpu().numpy(), axis=1)
            
            # 统计内点
            inlier_mask = errors < threshold
            inlier_count = np.sum(inlier_mask)
            
            # 更新最佳变换
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_H = H
                
        except Exception as e:
            # 处理估计失败的情况
            print(f"Iteration {i} failed: {e}")
            continue
    
    return best_H
def ransac_projective(X1_np, X2_np, threshold=0.01, max_iter=300, sample_size=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to torch tensors on GPU
    X1 = torch.tensor(X1_np, dtype=torch.float32, device=device)
    X2 = torch.tensor(X2_np, dtype=torch.float32, device=device)
    N = X1.shape[0]

    # Sample indices for each hypothesis.
    indices = torch.randint(0, N, (max_iter, sample_size), device=device)

    # Gather sampled point sets.
    X1_samples = torch.stack([X1[idx] for idx in indices])  # (max_iter, sample_size, 3)
    X2_samples = torch.stack([X2[idx] for idx in indices])  # (max_iter, sample_size, 3)

    # Estimate homographies.
    H_ests = estimate_3D_homography(X1_samples.cpu().numpy(), X2_samples.cpu().numpy())

    # Apply homographies to all points.
    X2_preds =  apply_homography_batch(H_ests, X1)

    # Compute Euclidean error.
    errors = torch.norm(X2_preds - X2[None, :, :], dim=2)

    # Compute inlier masks and counts.
    inlier_masks = errors < threshold  # (max_iter, N)
    inlier_counts = inlier_masks.sum(dim=1)

    # Select best hypothesis
    best_idx = torch.argmax(inlier_counts)
    best_H = H_ests[best_idx].cpu().numpy()

    return best_H
def turbo_reg(X1_np, X2_np):
    kpts_src = torch.from_numpy(X1_np)
    kpts_dst = torch.from_numpy(X2_np)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kpts_src = kpts_src.to(device).float()
    kpts_dst = kpts_dst.to(device).float()
    # Initialize TurboReg with specific parameters:
    reger = turboreg_gpu.TurboRegGPU(
        6000,      # max_N: Maximum number of correspondences
        0.012,     # tau_length_consis: \tau (consistency threshold for feature length/distance)
        2000,      # num_pivot: Number of pivot points, K_1
        0.15,      # radiu_nms: Radius for avoiding the instability of the solution
        0.1,       # tau_inlier: Threshold for inlier points. NOTE: just for post-refinement (REF@PointDSC/SC2PCR/MAC)
        "IN"       # eval_metric: MetricType (e.g., "IN" for Inlier Number, or "MAE" / "MSE")
    )

    # Run registration
    trans = reger.run_reg(kpts_src, kpts_dst).numpy()
    return trans
def icp(X1_np, X2_np, threshold=0.01, max_iter=300, tolerance=1e-6, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to torch tensors on GPU or CPU
    X1 = torch.tensor(X1_np, dtype=torch.float32, device=device)
    X2 = torch.tensor(X2_np, dtype=torch.float32, device=device)
    
    # Initialize
    prev_error = float('inf')
    
    # Iterative ICP loop
    for iter in range(max_iter):
        all_distances = []  # 用于存储所有批次的距离矩阵

        # Step 1: Find closest points in X2 for each point in X1, using batch processing
        num_points = X1.shape[0]
        num_batches = (num_points + batch_size - 1) // batch_size  # 计算总共的批次数量

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_points)
            
            X1_batch = X1[start_idx:end_idx]
            X2_batch = X2

            distances_batch = torch.cdist(X1_batch, X2_batch)
            all_distances.append(distances_batch)

        distances = torch.cat(all_distances, dim=0)


        closest_indices = torch.argmin(distances, dim=1)  
        X2_closest = X2[closest_indices] 

        centroid_X1 = X1.mean(dim=0)
        centroid_X2 = X2_closest.mean(dim=0)

        X1_centered = X1 - centroid_X1
        X2_centered = X2_closest - centroid_X2

        H = torch.matmul(X1_centered.t(), X2_centered)

        U, _, V = torch.svd(H)

        R = torch.matmul(V, U.t())

        t = centroid_X2 - torch.matmul(R, centroid_X1)

        X1_transformed = torch.matmul(X1, R.t()) + t

        error = torch.norm(X1_transformed - X2_closest, p=2)

        if abs(prev_error - error) < tolerance:
            print(f"ICP converged at iteration {iter}")
            break
        prev_error = error

        torch.cuda.empty_cache()

    return R, t
