import numpy as np
import matplotlib.pyplot as plt

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

def generate_grid(width, height, step):
    x = np.arange(0, width, step)
    y = np.arange(0, height, step)
    grid_points = np.array([(i, j) for j in y for i in x], dtype=np.float32)
    return grid_points

def plot_vectors(orig_pts, distorted_pts):
    plt.quiver(orig_pts[:, 0], orig_pts[:, 1],
               distorted_pts[:, 0] - orig_pts[:, 0],
               distorted_pts[:, 1] - orig_pts[:, 1],
               angles='xy', scale_units='xy', scale=1, color='r')

if __name__ == "__main__":
    # 기본 이미지 크기와 격자 간격 설정
    width, height = 640, 480
    step = 40

    # 카메라 파라미터 설정
    fx, fy = 800, 800  # 초점 거리
    cx, cy = width // 2, height // 2  # 이미지 중심
    k1, k2, k3 = -0.2, 0.1, 0.0  # 왜곡 계수 예시

    f = (fx, fy)
    c = (cx, cy)

    # 격자 생성
    grid = generate_grid(width, height, step)

    # 왜곡 적용
    distorted_grid = distort_points(grid, f, c, k1, k2, k3)

    # 벡터 시각화
    plt.figure(figsize=(10, 6))
    plot_vectors(grid, distorted_grid)
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.gca().invert_yaxis()
    plt.title('Grid Distortion Vector Visualization')
    plt.show()
