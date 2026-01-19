import cv2
import torch
import pycolmap
import numpy as np
from pathlib import Path
from models import FeatureExtractor

def create_map(video_path, output_dir="onepose_map"):
    device = torch.device("cuda")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # --- STEP 1: EXTRACT FRAMES ---
    cap = cv2.VideoCapture(video_path)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    extractor = FeatureExtractor().to(device)
    
    frame_list = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % 5 == 0:
            path = images_dir / f"{frame_idx:05d}.jpg"
            cv2.imwrite(str(path), frame)
            frame_list.append(frame)
        frame_idx += 1
    cap.release()

    # --- STEP 2: COLMAP SFM ---
    db_path = output_dir / "database.db"
    if db_path.exists(): db_path.unlink()
    pycolmap.extract_features(db_path, images_dir)
    pycolmap.match_exhaustive(db_path)
    recs = pycolmap.incremental_mapping(db_path, images_dir, output_dir / "sparse")
    
    if not recs: return print("SFM Failed. Move slower!")
    rec = recs[0]

    # --- STEP 3: MAPPING REAL FEATURES ---
    # This replaces the GAT for now to ensure a "Lock"
    map_pts_3d = []
    map_descriptors = []

    print("Building Neural Map from localized 3D points...")
    for p3d_id, p3d in rec.points3D.items():
        # Get the first image that saw this 3D point
        track_element = p3d.track.elements[0]
        image_id = track_element.image_id
        point2d_idx = track_element.point2D_idx
        
        # Get the actual image data
        img_info = rec.images[image_id]
        frame_bgr = cv2.imread(str(images_dir / img_info.name))
        
        # Extract features from that specific frame
        img_t = torch.from_numpy(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).float().permute(2,0,1)[None].to(device) / 255.0
        kp, des = extractor.run(img_t)
        
        # Find the descriptor closest to the COLMAP 2D point
        colmap_kp = img_info.points2D[point2d_idx].xy
        dist = torch.norm(kp - torch.tensor(colmap_kp).to(device), dim=1)
        best_idx = torch.argmin(dist)

        map_pts_3d.append(p3d.xyz)
        map_descriptors.append(des[best_idx].cpu().numpy())

    map_data = {
        "points_3d": np.array(map_pts_3d),
        "descriptors": np.array(map_descriptors)
    }
    torch.save(map_data, output_dir / "map.pt")
    print(f"Map Built: {len(map_pts_3d)} points.")

if __name__ == "__main__":
    create_map("input_video.mp4") 