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
    
def distort(image, f, c, k1, k2, k3, cam_mode = "KB"):
    fx, fy = f
    cx, cy = c  
    h, w = image.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = (x - cx) / fx
    y = (y - cy) / fy
    r = np.sqrt(x**2 + y**2)

    if cam_mode == "BC":
        x_distorted = x * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)
        y_distorted = y * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)
    elif cam_mode == "KB":
        # 4. 각도 계산
        theta = np.arctan(r)
        
        # 5. 왜곡된 반경 계산
        theta_distorted = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6)
        # 6. 왜곡된 좌표 계산 (정규화된 x, y 좌표)
        scale = np.where(r != 0, theta_distorted / r, 1)  # r이 0인 경우를 대비해 1로 설정
        x_distorted = x * scale
        y_distorted = y * scale
    
    x_distorted = (x_distorted * fx) + cx
    y_distorted = (y_distorted * fy) + cy
    map_x = np.float32(x_distorted)
    map_y = np.float32(y_distorted)
    distorted_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return distorted_image

def distort_points_CV(pts, f, c, k1, k2, k3):
    # Construct the camera matrix
    camera_matrix = np.array([[f[0], 0, c[0]],
                              [0, f[1], c[1]],
                              [0, 0, 1]], dtype=np.float64)
    
    # Set distortion coefficients
    dist_coeffs = np.array([k1, k2, 0, 0, k3], dtype=np.float64)
    
    # Convert points to the required shape
    pts = np.array(pts, dtype=np.float64).reshape(-1, 1, 2)
    
    # Apply distortion
    distorted_pts = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, None, camera_matrix)
    
    return distorted_pts.reshape(-1, 2)


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

def test_distort(image_file, output_file, f, c, cam_distort):
    # 이미지 로드
    image = cv2.imread(image_file)

    # 왜곡 계수 설정 (양수는 배럴 왜곡, 음수는 핀쿠션 왜곡)
    k1, k2, k3 = cam_distort

    # 왜곡 적용
    distorted_image = distort(image, f, c, k1, k2, k3)
    cv2.imwrite(output_file, distorted_image)
    print(f"Save distorted image in {output_file}")

    # 결과 이미지 보기
    # GUI 환경에서만 사용 가능
    cv2.imshow("Distorted Image", distorted_image)
    cv2.destroyAllWindows()

def test_distort_new(image_file, f, c, cam_distort, K, det_result_file):
    # 이미지 로드
    image = cv2.imread(image_file)

    # 왜곡 계수 설정 (양수는 배럴 왜곡, 음수는 핀쿠션 왜곡)
    k1, k2, k3, k4 = cam_distort

    # 왜곡 적용
    distorted_image1 = distort(image, f, c, k1, k2, k3, cam_mode ="KB")
    distorted_image2 = distort(image, f, c, k1, k2, k3, cam_mode ="BC")


    with open(det_result_file) as f_in:
        
        det_result = f_in.read().splitlines()[0]
        # e.g) 1,1,584.6,446.2,87.8,261.9,0.96,-1,-1,-1
        frame_id, id, bb_left, bb_top, bb_w, bb_h, confidence_score, x, y, z = det_result.split(',')

        # Plot original bbox 
        cv2.rectangle(image, (int(float(bb_left)), int(float(bb_top))), 
                     ((int(float(bb_left)) + int(float(bb_w))), (int(float(bb_top))+int(float(bb_h)))), (0, 255, 0), 2)

        # Distort points 
        pts = convert_raw_to_pts(int(float(bb_left)), int(float(bb_top)), int(float(bb_w)), int(float(bb_h)))        
        distorted_pts1 = distort_points_KB(pts, f, c, k1, k2, k3) 
        distorted_pts2 = distort_points_BC(pts, f, c, k1, k2, k3)

        # Ensure distorted_pts is a valid numpy array
        if distorted_pts1.size > 0:
            distorted_pts1 = np.array(distorted_pts1, dtype=np.float32)  
            distorted_pts2 = np.array(distorted_pts2, dtype=np.float32) 
            # Create new bounding box which is smallest rectangle including distorted points  
            u1, v1, w1, h1 = cv2.boundingRect(distorted_pts1)
            u2, v2, w2, h2 = cv2.boundingRect(distorted_pts2)
            # Draw the rectangle on the image
            cv2.rectangle(distorted_image1, (u1, v1), (u1 + w1, v1 + h1), (0, 0, 255), 2)
            cv2.rectangle(distorted_image2, (u2, v2), (u2 + w2, v2 + h2), (255, 0, 0), 2)
        
  
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.axis("off")  # Tắt các trục tọa độ
    plt.title("Image 1")

    # Vẽ ảnh 2
    plt.subplot(2, 2, 2)
    plt.imshow(distorted_image1)
    plt.axis("off")  # Tắt các trục tọa độ
    plt.title("Image KB")

    plt.subplot(2, 2, 3)
    plt.imshow(distorted_image2)
    plt.axis("off")  # Tắt các trục tọa độ
    plt.title("Image BC")

    plt.show()

def test_dist_pts(image_file, det_result_file, K, cam_distort):
    with open(det_result_file) as f_in:
        image = cv2.imread(image_file)
        
        image = distort_new(image, K, cam_distort)
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
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        

def create_distorted_mot17(gt_file, save_file, image_file, K, cam_dist):
    with open(gt_file) as f_in:
        gts = f_in.read().splitlines()

        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        with open(save_file, 'w') as out_file:
            for gt in tqdm(gts):
                frame_id, id, bb_left, bb_top, bb_w, bb_h, confidence_score, class_id, visibility = gt.split(',')
                pts = convert_raw_to_pts(int(float(bb_left)), int(float(bb_top)), int(float(bb_w)), int(float(bb_h)))

                image = cv2.imread(image_file)

                pt1, pt2, pt3, pt4 = pts
                cv2.rectangle(image, tuple(pt1), tuple(pt3), color=(0, 255, 0), thickness=2)
                cv2.putText(image, str(id), tuple(pt1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                distorted_pts = distort_points_KB(pts, K, cam_dist)

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
                
                # Use only first frame image to get width and height of image

                image = cv2.imread(image_file)
                distorted_pts = distort_points_new(pts, K, cam_dist)

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
        # params = [cam_focal, cam_center, dist_coeffs]
        dist_coeffs = np.array([0.5, 0.3 ,0.0, 0.0])
        params = [K, dist_coeffs]
        # test_dist_pts("/home/chanhoseo/motws/Data/MOT17/train/"+seq+"/img1/000001.jpg",
        #                 "det_results/mot17/"+seq+".txt",                      
        #                 *params)

        test_distort_new("/home/chanhoseo/motws/Data/MOT17/train/"+seq+"/img1/000001.jpg",
                        f, c, dist_coeffs, K,
                        "det_results/mot17/"+seq+".txt",)
        
        # # Make distorted image 
        # test_distort("/home/chanhoseo/motws/Data/MOT17/train/"+seq+"/img1/000001.jpg",
        #              '/home/chanhoseo/motws/Data/distortMOT17/images/'+seq+'_distorted_image.jpg',
        #             *params)

        # # Create distorted detection result 
        # create_distorted_det_results(
        #                         "det_results/mot17/"+seq+".txt", 
        #                         "/home/chanhoseo/motws/Data/DISTORTMOT17_val/"+seq+"/"+seq+".txt", 
        #                         "/home/chanhoseo/motws/Data/MOT17/train/"+seq+"/img1/000001.jpg", 
        #                         *params)

        # # Create distorted gt 
        # create_distorted_mot17(
        #                 '/home/chanhoseo/motws/Data/MOT17/train/'+seq+"/gt/gt.txt",
        #                 '/home/chanhoseo/motws/Data/DISTORTMOT17_val/'+seq+'/gt/gt.txt',
        #                 "/home/chanhoseo/motws/Data/MOT17/train/"+seq+"/img1/000001.jpg", 
        #                 *params)
