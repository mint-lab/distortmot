import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def readCamParaFile(camera_para, flag_KRT=False):
    R = np.zeros((3, 3))
    T = np.zeros((3, 1))
    IntrinsicMatrix = np.zeros((3, 3))
    try:
        with open(camera_para, 'r') as f_in:
            lines = f_in.readlines()
        i = 0
        while i < len(lines):
            if lines[i].strip() == "RotationMatrices":
                i += 1
                for j in range(3):
                    R[j] = np.array(list(map(float, lines[i].split())))
                    i += 1
            elif lines[i].strip() == "TranslationVectors":
                i += 1
                T = np.array(list(map(float, lines[i].split()))).reshape(-1, 1)
                T = T / 1000
                i += 1
            elif lines[i].strip() == "IntrinsicMatrix":
                i += 1
                for j in range(3):
                    IntrinsicMatrix[j] = np.array(list(map(float, lines[i].split())))
                    i += 1
            else:
                i += 1
    except FileNotFoundError:
        print(f"Error! {camera_para} doesn't exist.")
        return None, False

    Ki = np.zeros((3, 4))
    Ki[:, :3] = IntrinsicMatrix

    Ko = np.eye(4)
    Ko[:3, :3] = R
    Ko[:3, 3] = T.flatten()

    if flag_KRT:
        return IntrinsicMatrix, R, T.flatten(), True
    else:
        KiKo = np.dot(Ki, Ko)
        return Ki, Ko, True

def distort(image, f, c, k1, k2, k3):
    fx, fy = f
    cx, cy = c  
    dist_coeffs = np.array([k1, k2, 0, 0, k3], dtype=np.float64)

    K = np.array([[fx, 0, cx],
                 [0, fy, cy],
                 [0, 0, 1]])
    
    h, w = image.shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Normalize coordinates
    # u = (u - cx) / fx
    # v = (v - cy) / fy
    uv = np.stack((u, v), axis=-1).reshape(-1, 1, 2)  # 각 점의 (u, v) 좌표 쌍 생성 및 리쉐이프
    r = np.sqrt(u**2 + v**2)

    # Using undistortPoints to calculate distorted coordinates
    uv_distorted = cv2.undistortPoints(uv.astype(np.float32), K, dist_coeffs)
    uv_distorted = uv_distorted.reshape(h, w, 2)  # 원래 이미지 형태로 리쉐이프
    
    # 분할
    u_distorted = uv_distorted[:, :, 0]  # u 좌표
    v_distorted = uv_distorted[:, :, 1]  # v 좌표

    # Denormalize the coordinates
    u_distorted = (u_distorted * fx) + cx
    v_distorted = (v_distorted * fy) + cy
    map_x = np.float32(u_distorted)
    map_y = np.float32(v_distorted)
    distorted_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    return distorted_image

def distort_points_BC(pts, f, c, k1, k2, k3):
    fx, fy = f
    cx, cy = c
    
    # 변환된 포인트를 저장할 리스트
    distorted_pts = []

    for pt in pts:
        x, y = pt
        
        # 렌즈 중심 기준으로 좌표 정규화
        x_norm = (x - cx) / fx
        y_norm = (y - cy) / fy
        
        # 반경 계산
        r = np.sqrt(x_norm**2 + y_norm**2)
        
        # 방사 왜곡 적용
        x_distorted = x_norm * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)
        y_distorted = y_norm * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)
        
        # 원래 픽셀 좌표로 변환
        x_distorted = x_distorted * fx + cx
        y_distorted = y_distorted * fy + cy
        
        distorted_pts.append((x_distorted, y_distorted))
    
    return np.array(distorted_pts)

def distort_points_KB(pts, f, c, k1, k2, k3):
    fx, fy = f
    cx, cy = c
    
    # 변환된 포인트를 저장할 리스트
    distorted_pts = []

    for pt in pts:
        x, y = pt
        
        # 렌즈 중심 기준으로 좌표 정규화
        x_norm = (x - cx) / fx
        y_norm = (y - cy) / fy
        
        # 반경 계산
        r = np.sqrt(x_norm**2 + y_norm**2)
        
        # 각도 계산
        theta = np.arctan(r)
        
        # Kannala-Brandt 모델의 왜곡 적용
        theta_distorted = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6)
        
        # 왜곡된 반경을 사용해 정규화된 좌표 계산
        scale = np.where(r != 0, theta_distorted / r, 1)  # r이 0인 경우 scale을 1로 설정
        x_distorted = x_norm * scale
        y_distorted = y_norm * scale
        
        # 원래 픽셀 좌표로 변환
        x_distorted = x_distorted * fx + cx
        y_distorted = y_distorted * fy + cy
        
        distorted_pts.append((x_distorted, y_distorted))
    
    return np.array(distorted_pts)

def convert_raw_to_pts(bb_left, bb_top, bb_w, bb_h):
    """
    Convert uvwh to each point array
    """
    pt1 = np.array([bb_left,               bb_top])
    pt2 = np.array([bb_left + bb_w,        bb_top])
    pt3 = np.array([bb_left + bb_w, bb_top + bb_h])
    pt4 = np.array([bb_left,        bb_top + bb_h])
    return pt1, pt2, pt3, pt4 

def create_distorted_mot17(gt_file, save_file, image_file, K, cam_dist):
    with open(gt_file) as f_in:
        gts = f_in.read().splitlines()
        
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        with open(save_file, 'w') as out_file:
            for gt in tqdm(gts):
                frame_id, id, bb_left, bb_top, bb_w, bb_h, confidence_score, class_id, visibility = gt.split(',')
                pts = convert_raw_to_pts(int(float(bb_left)), int(float(bb_top)), int(float(bb_w)), int(float(bb_h)))
                
                # Distort 
                k1, k2, k3 = cam_dist[0], cam_dist[1], 0
                distorted_pts = distort_points_BC(pts, K, k1, k2, k3)

                # Ensure distorted_pts is a valid numpy array
                if distorted_pts.size > 0:
                    distorted_pts = np.array(distorted_pts, dtype=np.float32)  # Convert to float32 if necessary
                    
                    # Create new bounding box which is smallest rectangle including distorted points  
                    u, v, w, h = cv2.boundingRect(distorted_pts)
                    
                    # Write the results to file
                    out_file.write(f"{frame_id},{id},{u},{v},{w},{h},{confidence_score},{class_id},{visibility}\n")
                else:
                    print(f"Warning: No distorted points found for frame_id {frame_id} and id {id}")


def create_distorted_det_results(det_result_file, save_file, image_file, K, cam_dist):
    with open(det_result_file) as f_in:
        det_results = f_in.read().splitlines()
        
        with open(save_file, 'w') as out_file:
            for det_result in tqdm(det_results):
                # e.g) 1,1,584.6,446.2,87.8,261.9,0.96,-1,-1,-1
                frame_id, id, bb_left, bb_top, bb_w, bb_h, confidence_score, x, y, z = det_result.split(',')
                pts = convert_raw_to_pts(int(float(bb_left)), int(float(bb_top)), int(float(bb_w)), int(float(bb_h)))
                
                # Distort 
                k1, k2, k3 = cam_dist[0], cam_dist[1], 0
                distorted_pts = distort_points_BC(pts, K, k1, k2, k3)
                
                # Ensure distorted_pts is a valid numpy array
                if distorted_pts.size > 0:
                    distorted_pts = np.array(distorted_pts, dtype=np.float32)  # Convert to float32 if necessary
                    
                    # Create new bounding box which is smallest rectangle including distorted points  
                    u, v, w, h = cv2.boundingRect(distorted_pts)
                    
                    # Write the results to file
                    out_file.write(f"{frame_id},{id},{u},{v},{w},{h},{confidence_score},{x},{y},{z}\n")
                else:
                    print(f"Warning: No distorted points found for frame_id {frame_id} and id {id}")




if __name__ == "__main__":

    # new GT 
    sequences = ["MOT17-02-SDP", "MOT17-04-SDP", "MOT17-05-SDP","MOT17-09-SDP",
                 "MOT17-10-SDP","MOT17-11-SDP","MOT17-13-SDP"]
    
    for seq in sequences:    

        # Get Intrinsic Matrix from CamParafile 
        K,_,_ = readCamParaFile("cam_para/MOT17/"+seq+".txt")
        
        if K.shape[1] >3:
            K = K[:, :3] # Make sure 3x3
        K = K.astype(np.float32)
        f = (K[0][0], K[1][1])
        c = (K[0][2], K[1][2])
        
        dist_coeffs = np.array([-0.5, 0.3 ,0.0, 0.0])
        params = [cam_focal, cam_center, dist_coeffs]

        # Create distorted detection result 
        create_distorted_det_results(
                                "det_results/mot17/"+seq+".txt", 
                                "/home/chanhoseo/motws/Data/DISTORTMOT17_val/"+seq+"/"+seq+".txt", 
                                "/home/chanhoseo/motws/Data/MOT17/train/"+seq+"/img1/000001.jpg", 
                                *params)

        # Create distorted gt 
        create_distorted_mot17(
                        '/home/chanhoseo/motws/Data/MOT17/train/'+seq+"/gt/gt.txt",
                        '/home/chanhoseo/motws/Data/DISTORTMOT17_val/'+seq+'/gt/gt.txt',
                        "/home/chanhoseo/motws/Data/MOT17/train/"+seq+"/img1/000001.jpg", 
                        *params)
