import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import time

def main():
    try:
        print("[TEST] Setting up OffscreenRenderer...")
        render = rendering.OffscreenRenderer(640, 480)
        
        print(f"[TEST] Loading GLB model...")
        # Add model
        render.scene.add_model("chick", rendering.MaterialRecord(), o3d.io.read_triangle_model("model/animated_chick.glb"))
        
        # Camera
        render.setup_camera(60.0, [0,0,0], [0,0,5], [0,1,0])
        
        # Render
        img = np.asarray(render.render_to_image())
        print(f"[TEST] Render successful, image shape: {img.shape}")
        
    except Exception as e:
        print(f"[ERROR] Failed to render: {e}")

if __name__ == '__main__':
    main()
