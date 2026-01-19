import cv2
import torch
import numpy as np
import kornia.feature as KF
import time
from models import FeatureExtractor 

# --- JITTER REDUCTION FILTER ---
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0
        self.t_prev = None

    def __call__(self, x):
        t = time.time()
        if self.t_prev is None:
            self.t_prev, self.x_prev = t, x
            return x
        te = t - self.t_prev
        ad = 1.0 / (1.0 + 2 * np.pi * te * self.d_cutoff)
        dx = (x - self.x_prev) / te
        edx = ad * dx + (1 - ad) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        a = 1.0 / (1.0 + 2 * np.pi * te * cutoff)
        out = a * x + (1 - a) * self.x_prev
        self.t_prev, self.x_prev, self.dx_prev = t, out, edx
        return out

def frame_to_tensor(frame, device):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).float().permute(2, 0, 1)[None] / 255.0
    return img.to(device)

def run_inference(map_path="onepose_map/map.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. LOAD MAP
    print(f"Loading Neural Map: {map_path}")
    map_data = torch.load(map_path, weights_only=False)
    map_pts_3d = torch.from_numpy(map_data["points_3d"]).to(device).float()
    map_descs = torch.from_numpy(map_data["descriptors"]).to(device).float()
    map_pts_2d_dummy = torch.zeros((1, map_pts_3d.shape[0], 2)).to(device)

    # 2. SETUP NETWORKS
    extractor = FeatureExtractor().to(device)
    matcher = KF.LightGlue(features='disk').to(device).eval()
    
    # 3. INITIALIZE FILTERS
    filter_r = OneEuroFilter(min_cutoff=0.5, beta=0.005) # Rotation smoother
    filter_t = OneEuroFilter(min_cutoff=0.1, beta=0.01)  # Translation smoother

    cap = cv2.VideoCapture(0)
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    print("--- LIVE INFERENCE ACTIVE ---")

    while True:
        ret, frame = cap.read()
        if not ret: break
        h, w = frame.shape[:2]
        
        img_t = frame_to_tensor(frame, device)
        kp_live, des_live = extractor.run(img_t)
        
        input_dict = {
            "image0": {"keypoints": map_pts_2d_dummy, "descriptors": map_descs.unsqueeze(0), "image_size": torch.tensor([[w, h]], device=device)},
            "image1": {"keypoints": kp_live.unsqueeze(0), "descriptors": des_live.unsqueeze(0), "image_size": torch.tensor([[w, h]], device=device)}
        }
        
        with torch.no_grad():
            matches_out = matcher(input_dict)
            m0 = matches_out['matches0'][0] 
            valid_mask = m0 > -1
            idx_map = torch.where(valid_mask)[0]
            idx_live = m0[valid_mask]
            
        if idx_map.shape[0] > 12:
            try:
                p3d = map_pts_3d[idx_map].cpu().numpy().astype(np.float32)
                p2d = kp_live[idx_live].cpu().numpy().astype(np.float32)
                
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    p3d.reshape(-1,1,3), p2d.reshape(-1,1,2), K, None, 
                    iterationsCount=100, reprojectionError=6.0, flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success and inliers is not None:
                    # Apply Smoothing Filter
                    rvec = filter_r(rvec)
                    tvec = filter_t(tvec)

                    # Dynamic Axis Placement
                    center_3d = np.mean(p3d[inliers.flatten()], axis=0)
                    extent = np.linalg.norm(np.max(p3d, axis=0) - np.min(p3d, axis=0))
                    axis_size = extent * 0.4
                    
                    axis_pts = np.float32([center_3d, center_3d + [axis_size,0,0], 
                                           center_3d + [0,axis_size,0], center_3d + [0,0,axis_size]])
                    
                    imgpts, _ = cv2.projectPoints(axis_pts, rvec, tvec, K, None)
                    pts = imgpts.reshape(-1, 2).astype(int)
                    
                    # Draw Axis
                    origin = tuple(pts[0])
                    cv2.line(frame, origin, tuple(pts[1]), (0,0,255), 3) # X
                    cv2.line(frame, origin, tuple(pts[2]), (0,255,0), 3) # Y
                    cv2.line(frame, origin, tuple(pts[3]), (255,0,0), 3) # Z
                    
                    # Draw Feature Dots (Visual feedback)
                    for i in inliers.flatten():
                        cv2.circle(frame, tuple(p2d[i].astype(int)), 2, (0, 255, 0), -1)

                    cv2.putText(frame, f"LOCK: {len(inliers)}", (10, 40), 2, 0.7, (0,255,0), 2)
            except Exception: pass

        cv2.imshow("OnePose-Lite Smooth", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference()