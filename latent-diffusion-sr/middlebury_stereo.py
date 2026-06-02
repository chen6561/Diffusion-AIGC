import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ====================== 1. 全局设置：解决中文显示 ======================
plt.rcParams['font.sans-serif'] = ['SimHei']   # 设置中文字体，解决中文乱码
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示异常问题

# ====================== 2. 路径配置 ======================
# 数据集根目录（你自己的路径，不用改）
DATASET_ROOT = r"D:/datasets/Stereo Vision/Middlebury"
# 结果图片保存目录，程序会自动创建
SAVE_ROOT = "./stereo_results"
os.makedirs(SAVE_ROOT, exist_ok=True)  # 自动创建文件夹，不存在则创建

# ====================== 3. 要处理的场景列表 ======================
# 这些是 Middlebury2006 标准双目数据集，只保留 -2 标准双目对
SCENES = [
    "conesF-ppm-2/conesF",
    "conesH-ppm-2/conesH",
    "conesQ-ppm-2/conesQ",
    "teddyF-ppm-2/teddyF",
    "teddyH-ppm-2/teddyH",
    "teddyQ-ppm-2/teddyQ"
]

# ====================== 4. 工具函数：判断图像分辨率 ======================
def get_res_type(scene_path):
    """
    从路径中判断当前场景是 F/H/Q 哪种分辨率
    F: Full 全分辨率
    H: Half 半分辨率
    Q: Quarter 1/4分辨率
    """
    if "F" in scene_path:
        return "F"
    elif "H" in scene_path:
        return "H"
    else:
        return "Q"

# ====================== 5. SGBM 立体匹配算法参数配置 ======================
def get_sgbm_params(res_type):
    """
    根据不同分辨率返回最优 SGBM 参数
    专门针对 Middlebury cones / teddy 数据集优化
    """
    # 该数据集视差不从0开始，整体偏移 20 个像素（关键！）
    min_disp = 20

    # 根据分辨率设置视差范围和匹配窗口
    if res_type == "F":
        num_disp = 256    # 视差搜索范围：必须是16的倍数
        block_size = 7    # 匹配窗口大小：必须是奇数
    elif res_type == "H":
        num_disp = 192
        block_size = 5
    else:
        num_disp = 128
        block_size = 3

    # P1/P2 控制视差平滑度，P2 > P1，官方推荐公式
    P1 = 8 * 3 * (block_size ** 2)
    P2 = 32 * 3 * (block_size ** 2)

    # 返回SGBM参数字典
    return {
        "minDisparity": min_disp,         # 最小视差值
        "numDisparities": num_disp,       # 视差搜索范围
        "blockSize": block_size,          # 匹配窗口大小
        "P1": P1,                         # 平滑参数1
        "P2": P2,                         # 平滑参数2
        "disp12MaxDiff": 2,               # 左右视差最大允许差异
        "uniquenessRatio": 15,            # 唯一性检测比例
        "speckleWindowSize": 200,         # 噪点过滤窗口
        "speckleRange": 4,                # 噪点允许范围
        "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY  # 算法模式
    }

# ====================== 6. 读取双目数据 + 真实视差图 ======================
def load_scene(scene_path):
    """
    读取一个场景的：左图、右图、真实视差图
    Middlebury2006 标准命名规则：
    im6.ppm → 左相机图像
    im2.ppm → 右相机图像
    disp2.pgm → 亚像素级真实视差图
    """
    folder = os.path.join(DATASET_ROOT, scene_path)
    print(f"正在读取文件夹：{folder}")

    # 拼接文件路径
    left_path = os.path.join(folder, "im6.ppm")
    right_path = os.path.join(folder, "im2.ppm")
    gt_path = os.path.join(folder, "disp2.pgm")

    # 判断文件是否存在
    if not all(os.path.exists(p) for p in [left_path, right_path, gt_path]):
        print("❌ 缺失文件，跳过该场景")
        return None

    # 读取图像
    left_img = np.array(Image.open(left_path))    # 左图
    right_img = np.array(Image.open(right_path))  # 右图
    gt_disparity = np.array(Image.open(gt_path)).astype(np.float32) / 16.0  # 真实视差：必须除以16才是真实值

    # 转为灰度图（立体匹配只需要灰度图）
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)

    print(f"✅ 读取成功：图像尺寸 {left_img.shape[1]}×{left_img.shape[0]}")
    return left_img, right_img, left_gray, right_gray, gt_disparity

# ====================== 7. 计算视差图 + 后处理 ======================
def compute_disparity(left_gray, right_gray, params):
    """
    使用 SGBM 计算视差图，并做后处理：
    1. 正向匹配：左图 → 右图
    2. 反向匹配：右图 → 左图
    3. 左右一致性检查：过滤错误匹配
    4. 中值滤波：去噪平滑
    """
    h, w = left_gray.shape
    min_disp = params["minDisparity"]
    num_disp = params["numDisparities"]

    # ========== 步骤1：左图 → 右图 计算视差 ==========
    stereo = cv2.StereoSGBM_create(**params)
    disp_left = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0  # 转为真实视差

    # ========== 步骤2：右图 → 左图 反向计算视差 ==========
    rev_params = params.copy()
    rev_params["minDisparity"] = -num_disp
    stereo_rev = cv2.StereoSGBM_create(**rev_params)
    disp_right = stereo_rev.compute(right_gray, left_gray).astype(np.float32) / 16.0

    # ========== 步骤3：左右一致性校验（去除错误匹配） ==========
    for y in range(h):
        for x in range(w):
            d = disp_left[y, x]
            # 过滤超出范围的视差
            if d < min_disp or d > min_disp + num_disp:
                disp_left[y, x] = 0
                continue
            # 找到对应点并检查一致性
            x_r = int(x - d)
            if 0 <= x_r < w and abs(d + disp_right[y, x_r]) > 1.5:
                disp_left[y, x] = 0

    # ========== 步骤4：中值滤波去噪 ==========
    disp_left = cv2.medianBlur(disp_left, 3)
    return disp_left

# ====================== 8. 评估视差精度 ======================
def evaluate(pred_disparity, gt_disparity):
    """
    计算三个核心评估指标：
    1. MAE：平均绝对误差
    2. RMSE：均方根误差
    3. Bad Pixel：误差>1像素的比例（立体匹配核心指标）
    只计算有效视差区域（gt>0）
    """
    mask = gt_disparity > 0  # 只评估有真值的区域
    if np.sum(mask) == 0:
        return 0, 0, 100.0

    # 计算误差
    error = np.abs(pred_disparity[mask] - gt_disparity[mask])
    mae = np.mean(error)                          # 平均误差
    rmse = np.sqrt(np.mean(error ** 2))           # 均方根误差
    bad_pixel = np.mean(error > 1.0) * 100        # 错误像素比例

    return mae, rmse, bad_pixel

# ====================== 9. 可视化结果并保存图片 ======================
def visualize(scene_name, left_img, right_img, gt_disparity, pred_disparity, metrics):
    """
    绘制 2行3列 结果图：
    1. 左图
    2. 右图
    3. 真实视差图
    4. 预测视差图
    5. 误差热力图
    6. 误差分布直方图
    自动保存到 stereo_results 文件夹
    """
    mae, rmse, bad = metrics
    plt.figure(figsize=(16, 8))

    # 左图
    plt.subplot(2, 3, 1)
    plt.imshow(left_img)
    plt.title("左图像 (im6.ppm)")
    plt.axis("off")

    # 右图
    plt.subplot(2, 3, 2)
    plt.imshow(right_img)
    plt.title("右图像 (im2.ppm)")
    plt.axis("off")

    # 真实视差图
    plt.subplot(2, 3, 3)
    plt.imshow(gt_disparity, cmap="jet")
    plt.title("真实视差图 (disp2.pgm)")
    plt.colorbar(label="视差 (像素)")
    plt.axis("off")

    # 预测视差图
    plt.subplot(2, 3, 4)
    plt.imshow(pred_disparity, cmap="jet")
    plt.title(f"SGBM 预测视差图\nMAE = {mae:.2f} 像素")
    plt.colorbar(label="视差 (像素)")
    plt.axis("off")

    # 误差热力图
    plt.subplot(2, 3, 5)
    error_map = np.abs(pred_disparity - gt_disparity)
    error_map[gt_disparity == 0] = 0
    plt.imshow(error_map, cmap="hot", vmin=0, vmax=3)
    plt.title(f"误差热力图\nRMSE = {rmse:.2f} 像素")
    plt.colorbar(label="误差 (像素)")
    plt.axis("off")

    # 误差直方图
    plt.subplot(2, 3, 6)
    mask = gt_disparity > 0
    plt.hist(np.abs(pred_disparity[mask] - gt_disparity[mask]), bins=50, range=(0, 3))
    plt.title("误差分布直方图")
    plt.xlabel("误差 (像素)")
    plt.ylabel("像素数量")

    plt.tight_layout()

    # 保存图片
    save_name = scene_name.replace("/", "_")
    save_path = os.path.join(SAVE_ROOT, f"{save_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ 图片已保存：{save_path}")
    plt.close()

# ====================== 10. 主函数：批量处理所有场景 ======================
if __name__ == "__main__":
    # 遍历所有场景
    for scene in SCENES:
        print(f"\n{'='*60}")
        print(f"正在处理场景：{scene}")
        print('='*60)

        # 读取数据
        data = load_scene(scene)
        if not data:
            continue

        # 解包数据
        left_img, right_img, left_gray, right_gray, gt_disp = data
        res = get_res_type(scene)          # 获取分辨率类型
        params = get_sgbm_params(res)       # 获取SGBM参数

        # 计算视差
        pred_disp = compute_disparity(left_gray, right_gray, params)

        # 评估精度
        mae, rmse, bad = evaluate(pred_disp, gt_disp)

        # 打印结果
        print(f"📊 评估结果：")
        print(f"  平均绝对误差(MAE): {mae:.2f} 像素")
        print(f"  均方根误差(RMSE): {rmse:.2f} 像素")
        print(f"  错误率(>1像素): {bad:.2f} %")

        # 显示并保存结果图
        visualize(scene, left_img, right_img, gt_disp, pred_disp, (mae, rmse, bad))