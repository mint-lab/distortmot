import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from distort import *

def distort_points_new(pts, K, dist_coeffs):
    if type(pts) != np.ndarray:
        pts = np.array(pts, dtype = np.float32)
    # Convert points to the required shape
    pts = cv2.convertPointsToHomogeneous(pts) # (N, 1, 3)
    Kinv = np.linalg.inv(K)
    pts = np.tensordot(pts, Kinv, axes=(2, 1))  # To camera coordinates, Shape: (N, 1, 3)
    pts = cv2.convertPointsFromHomogeneous(pts)  # Shape: (N, 1, 2)
    # Apply distortion (assumes intrinsic matrix of undistorted points is identity)
    distorted_pts = cv2.fisheye.distortPoints(pts, K = K, D = dist_coeffs)
    return distorted_pts

def test_fisheye(image_file, det_result_file, K, cam_distort, f, c):
    with open(det_result_file) as f_in:
        image = cv2.imread(image_file)
        k1, k2 , k3 = cam_distort[0], cam_distort[1], 0
        
        image = distort(image, f, c, k1, k2, k3)
        det_result = f_in.read().splitlines()[0]
        # e.g) 1,1,584.6,446.2,87.8,261.9,0.96,-1,-1,-1
        frame_id, id, bb_left, bb_top, bb_w, bb_h, confidence_score, x, y, z = det_result.split(',')

        # Plot original bbox 
        cv2.rectangle(image, (int(float(bb_left)), int(float(bb_top))), 
                    ((int(float(bb_left)) + int(float(bb_w))), (int(float(bb_top))+int(float(bb_h)))), (0, 255, 0), 2)

        # Distort points 
        pts = convert_raw_to_pts(int(float(bb_left)), int(float(bb_top)), int(float(bb_w)), int(float(bb_h)))        
        distorted_pts = distort_points_new(pts, K, cam_distort)

        # Ensure distorted_pts is a valid numpy array
        if distorted_pts.size > 0:
            distorted_pts = np.array(distorted_pts, dtype=np.float32)  # Convert to float32 if necessary
            # Create new bounding box which is smallest rectangle including distorted points  
            u, v, w, h = cv2.boundingRect(distorted_pts)
            
            # Draw the rectangle on the image
            cv2.rectangle(image, (u, v), (u + w, v + h), (0, 0, 255), 2)
            
            # Display the image with the rectangle
            cv2.imshow('Image with Bounding Box', image)
            key = cv2.waitKey(0) & 0xFF 
            if key == ord('q') or key == 27: # 'q' 이거나 'esc' 이면 종료
                
                cv2.destroyAllWindows()
    



if __name__ == "__main__":

    # new GT 
    sequences = ["MOT17-09-SDP"]
    
    for seq in sequences:    

        # Get Intrinsic Matrix from CamParafile 
        K,_,_ = readCamParaFile("cam_para/MOT17/"+seq+".txt")
        
        if K.shape[1] >3:
            K = K[:, :3] # Make sure 3x3
        K = K.astype(np.float32)
        f = (K[0][0], K[1][1])
        c = (K[0][2], K[1][2])
        
        # params = [cam_focal, cam_center, dist_coeffs]
        dist_coeffs = np.array([-0.5, 0.3 ,0.0, 0.0])
        params = [K, dist_coeffs]

        test_fisheye("/home/chanhoseo/motws/Data/MOT17/train/"+seq+"/img1/000001.jpg",
                        "det_results/mot17/"+seq+".txt",
                        K, dist_coeffs, f, c)
        
