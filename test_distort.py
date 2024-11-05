import numpy as np
import matplotlib.pyplot as plt
from distort import * 

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
    distorted_image1 = distort(image, f, c, k1, k2, k3)


    with open(det_result_file) as f_in:
        
        det_result = f_in.read().splitlines()[0]
        # e.g) 1,1,584.6,446.2,87.8,261.9,0.96,-1,-1,-1
        frame_id, id, bb_left, bb_top, bb_w, bb_h, confidence_score, x, y, z = det_result.split(',')

        # Plot original bbox 
        cv2.rectangle(image, (int(float(bb_left)), int(float(bb_top))), 
                     ((int(float(bb_left)) + int(float(bb_w))), (int(float(bb_top))+int(float(bb_h)))), (0, 255, 0), 2)

        # Distort points 
        pts = convert_raw_to_pts(int(float(bb_left)), int(float(bb_top)), int(float(bb_w)), int(float(bb_h)))        
        distorted_pts1 = distort_points_BC(pts, f, c, k1, k2, k3) 

        # Ensure distorted_pts is a valid numpy array
        if distorted_pts1.size > 0:
            distorted_pts1 = np.array(distorted_pts1, dtype=np.float32) 

            # Create new bounding box which is smallest rectangle including distorted points  
            u1, v1, w1, h1 = cv2.boundingRect(distorted_pts1)

            # Draw the rectangle on the image
            cv2.rectangle(distorted_image1, (u1, v1), (u1 + w1, v1 + h1), (255, 0, 0), 2)
        
  
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.axis("off")  # Tắt các trục tọa độ
    plt.title("Image 1")

    # Vẽ ảnh 2
    plt.subplot(2, 2, 2)
    plt.imshow(distorted_image1)
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
        dist_coeffs = np.array([-0.5, 0.3 ,0.0, 0.0])
        params = [K, dist_coeffs]

        test_distort_new("/home/chanhoseo/motws/Data/MOT17/train/"+seq+"/img1/000001.jpg",
                        f, c, dist_coeffs, K,
                        "det_results/mot17/"+seq+".txt")
        