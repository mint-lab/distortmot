import cv2
import numpy as np
import os
from tqdm import tqdm

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
    h, w = image.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = (x - cx) / fx
    y = (y - cy) / fy
    r = np.sqrt(x**2 + y**2)
    x_distorted = x * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)
    y_distorted = y * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)
    x_distorted = (x_distorted * fx) + cx
    y_distorted = (y_distorted * fy) + cy
    map_x = np.float32(x_distorted)
    map_y = np.float32(y_distorted)
    distorted_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return distorted_image

def distort_points(pts, f, c, k1, k2, k3):
    fx, fy = f 
    cx, cy = c
    pts_array = np.array(pts, dtype=np.float64)
    pts_normalized = np.zeros_like(pts_array)
    pts_normalized[:, 0] = (pts_array[:, 0] - cx) / fx
    pts_normalized[:, 1] = (pts_array[:, 1] - cy) / fy
    r = np.sqrt(pts_normalized[:, 0]**2 + pts_normalized[:, 1]**2)
    factor = 1 + k1 * r**2 + k2 * r**4 + k3 * r**6
    pts_distorted_normalized = pts_normalized * factor[:, np.newaxis]
    pts_distorted = np.zeros_like(pts_array)
    pts_distorted[:, 0] = (pts_distorted_normalized[:, 0] * fx) + cx
    pts_distorted[:, 1] = (pts_distorted_normalized[:, 1] * fy) + cy
    return pts_distorted

def convert_raw_to_bbox(bb_left, bb_top, bb_w, bb_h):
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

def create_distorted_mot17(gt_file, save_file, image_file, f, c, cam_distort):
    with open(gt_file) as f_in:
        gts = f_in.read().splitlines()
        k1, k2, k3 = cam_distort

        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        with open(save_file, 'w') as out_file:
            for gt in tqdm(gts):
                frame_id, id, bb_left, bb_top, bb_w, bb_h, confidence_score, class_id, visibility = gt.split(',')
                pts = convert_raw_to_bbox(int(float(bb_left)), int(float(bb_top)), int(float(bb_w)), int(float(bb_h)))

                image = cv2.imread(image_file)

                pt1, pt2, pt3, pt4 = pts
                cv2.rectangle(image, tuple(pt1), tuple(pt3), color=(0, 255, 0), thickness=2)
                cv2.putText(image, str(id), tuple(pt1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                distorted_pts = distort_points(pts, f, c, k1, k2, k3)

                # Ensure distorted_pts is a valid numpy array
                if distorted_pts.size > 0:
                    distorted_pts = np.array(distorted_pts, dtype=np.float32)  # Convert to float32 if necessary
                    
                    # Create new bounding box which is smallest rectangle including distorted points  
                    u, v, w, h = cv2.boundingRect(distorted_pts)
                    
                    # Write the results to file
                    out_file.write(f"{frame_id},{id},{u},{v},{w},{h},{confidence_score},{class_id},{visibility}\n")
                else:
                    print(f"Warning: No distorted points found for frame_id {frame_id} and id {id}")


def create_distorted_det_results(det_result_file, save_file, image_file, f, c, cam_distort):
    with open(det_result_file) as f_in:
        det_results = f_in.read().splitlines()
        k1, k2, k3 = cam_distort
        
        with open(save_file, 'w') as out_file:
            for det_result in tqdm(det_results):
                # e.g) 1,1,584.6,446.2,87.8,261.9,0.96,-1,-1,-1
                frame_id, id, bb_left, bb_top, bb_w, bb_h, confidence_score, x, y, z = det_result.split(',')

                pts = convert_raw_to_bbox(int(float(bb_left)), int(float(bb_top)), int(float(bb_w)), int(float(bb_h)))
                
                # Use only first frame image to get width and height of image

                image = cv2.imread(image_file)
                distorted_pts = distort_points(pts, f, c, k1, k2, k3)

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
    # 특정 이미지를 왜곡하여 확인
    # test_distort("img1/000001.jpg")

    # 특정 좌표를 왜곡하여 확인
    # test_distort_points()

    # new GT 
    sequences = ["MOT17-02-SDP", "MOT17-04-SDP", "MOT17-05-SDP","MOT17-09-SDP",
                 "MOT17-10-SDP","MOT17-11-SDP","MOT17-13-SDP"]
    
    cam_dist   = (0.2, 0.1, 0.0)
    for seq in sequences:    

        # Get focal length 
        K,_,_ = readCamParaFile("cam_para/MOT17/"+seq+".txt")
        cam_focal  = (K[0][0], K[1][1])
        cam_center = (K[0][2], K[1][2])
        params = [cam_focal, cam_center, cam_dist]

        # Make distorted image 
        test_distort("/home/chanhoseo/motws/Data/MOT17/train/"+seq+"/img1/000001.jpg",
                     '/home/chanhoseo/motws/Data/distortMOT17/images/'+seq+'_distorted_image.jpg',
                    *params)

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
