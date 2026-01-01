import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import cv2
from PIL import Image, ImageEnhance
from pathlib import Path
from ultralytics import YOLO  # YOLOv8核心库
import textwrap

# ===================== 全局配置（密集车辆场景专用） =====================
# 车辆图片目录（原始字符串避免转义）
CAR_IMG_DIR = r"F:\bird--CNN\data.cars"
# 支持的图片格式
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
# YOLOv8车辆类别映射（COCO数据集标签）
VEHICLE_LABELS = {
    2: 'car',  # 小汽车
    7: 'truck',  # 卡车
    3: 'motorcycle',  # 摩托车
    5: 'bus',  # 公交车
    6: 'train'  # 火车（按需删除）
}
# 密集车辆检测参数（核心优化）
DETECT_PARAMS = {
    "conf": 0.2,  # 降低置信度阈值（减少密集场景漏检）
    "iou": 0.3,  # 降低IoU阈值（避免过滤相邻车辆）
    "imgsz": 1280,  # 大尺寸输入（提升小目标车辆识别）
    "max_det": 300  # 最大检测数量（适配密集车流）
}
# 图片增强开关（密集场景建议开启）
ENHANCE_IMAGE = True
# 单张展示配置
SINGLE_DISPLAY_CONFIG = {
    "figsize": (16, 10),  # 单张图片展示窗口大小
    "font_size": 12,  # 文字大小
    "title_font_size": 14,  # 标题大小
    "close_after_show": True  # 关闭窗口后继续下一张
}

# ===================== 初始化配置 =====================
# 忽略冗余警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# 中文显示配置（修复乱码）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']  # 增加备选字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['figure.constrained_layout.use'] = True  # 自动调整布局避免裁剪
plt.rcParams['font.family'] = 'sans-serif'  # 明确字体族
# 设置Matplotlib后端为TkAgg（弹窗显示更友好）
plt.switch_backend('TkAgg')

# ===================== 加载YOLOv8x模型（高精度版） =====================
# 自动下载预训练权重（首次运行需联网）
model = YOLO('yolov8x.pt')
# 设置设备（有GPU用GPU，无则自动用CPU）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"✅ 模型加载完成 | 使用设备：{device}")


# ===================== 核心工具函数 =====================
def enhance_image(img):
    """图片增强：提升对比度/亮度/锐度（适配密集/低质车辆图片）"""
    # 对比度增强（突出车辆轮廓）
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    # 亮度增强（适配逆光/夜间场景）
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.2)
    # 锐度增强（清晰化车辆细节）
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.3)
    return img


def get_all_car_images():
    """获取目录下所有有效车辆图片路径（过滤损坏文件）"""
    img_paths = []
    for file in Path(CAR_IMG_DIR).rglob("*"):
        if file.suffix.lower() in SUPPORTED_FORMATS:
            # 校验文件是否可正常读取
            try:
                with Image.open(file) as f:
                    f.verify()  # 验证图片完整性
                img_paths.append(str(file))
            except Exception as e:
                print(f"⚠️ 跳过损坏文件：{file} | 原因：{str(e)[:50]}")

    if len(img_paths) == 0:
        raise FileNotFoundError(f"在目录 {CAR_IMG_DIR} 中未找到有效图片文件")

    print(f"✅ 共找到 {len(img_paths)} 张有效车辆图片")
    return img_paths


def preprocess_image(image_path):
    """图片预处理：加载+增强+格式转换"""
    try:
        # 加载图片并转换为RGB（统一格式）
        img = Image.open(image_path).convert('RGB')
        # 图片增强（可选）
        if ENHANCE_IMAGE:
            img = enhance_image(img)
        # 转换为numpy数组（适配YOLOv8输入）
        img_np = np.array(img)
        return img, img_np
    except Exception as e:
        print(f"⚠️ 加载/预处理图片失败：{image_path} | 原因：{str(e)[:50]}")
        return None, None


def detect_vehicles_single(image_path):
    """单张图片车辆检测（适配密集场景）"""
    raw_img, img_np = preprocess_image(image_path)
    if raw_img is None:
        return None, None, None, None, None

    # YOLOv8密集目标检测
    results = model.predict(
        source=img_np,
        conf=DETECT_PARAMS["conf"],
        iou=DETECT_PARAMS["iou"],
        imgsz=DETECT_PARAMS["imgsz"],
        max_det=DETECT_PARAMS["max_det"],
        device=device,
        verbose=False  # 关闭单张图片的冗余日志
    )[0]  # 取第一张图的检测结果

    # 统计车辆类别和总数
    vehicle_count = 0
    vehicle_detail = {v_name: 0 for v_name in VEHICLE_LABELS.values()}
    for det in results.boxes:
        cls = int(det.cls[0])  # 获取检测类别
        conf = float(det.conf[0])  # 获取置信度
        # 仅统计目标车辆类别+置信度达标
        if cls in VEHICLE_LABELS and conf >= DETECT_PARAMS["conf"]:
            v_name = VEHICLE_LABELS[cls]
            vehicle_detail[v_name] += 1
            vehicle_count += 1

    img_name = os.path.basename(image_path)
    return results, vehicle_count, vehicle_detail, img_name, raw_img


def create_result_text(count, detail, img_name, max_width=40):
    """生成纯文字格式化检测结果（移除特殊符号）"""
    # 格式化图片名称（自动换行）
    wrapped_name = textwrap.fill(f"图片名称：{img_name}", max_width)

    # 构建结果文本（仅用纯文字，移除所有特殊符号）
    text_lines = [
        wrapped_name,
        f"检测到车辆总数：{count}",
        "---------------------------",
        "车辆类别详细分布："
    ]

    # 添加各类别统计（仅显示有数量的）
    for v_type, v_count in detail.items():
        if v_count > 0:
            ratio = v_count / count * 100 if count > 0 else 0
            text_lines.append(f"• {v_type}：{v_count} 辆（占比 {ratio:.1f}%）")

    # 补充无检测结果的情况
    if count == 0:
        text_lines.append("• 未检测到任何车辆")

    # 合并为带换行符的文本
    return "\n".join(text_lines)


def display_single_image_result(results, vehicle_count, vehicle_detail, img_name, raw_img):
    """单张图片结果独立弹窗展示（左图右文）"""
    # 创建新的画布（每次都新建，避免缓存）
    fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=SINGLE_DISPLAY_CONFIG["figsize"],
                                          gridspec_kw={'width_ratios': [2, 1]})

    # 设置窗口标题
    fig.canvas.manager.set_window_title(f"车辆检测结果 - {img_name}")

    # 设置总标题
    fig.suptitle(
        f"密集车辆检测结果（置信度≥{DETECT_PARAMS['conf']}）",
        fontsize=SINGLE_DISPLAY_CONFIG["title_font_size"],
        fontweight='bold',
        y=0.95
    )

    # 1. 绘制检测后的图片
    render_img = results.plot(
        conf=True,  # 显示置信度
        labels=True,  # 显示类别标签
        line_width=2,  # 加粗检测框（密集场景更清晰）
        font_size=8,  # 缩小标签字体，避免遮挡
        pil=True  # 返回PIL图片
    )
    ax_img.imshow(render_img)
    ax_img.axis('off')  # 关闭图片坐标轴
    ax_img.set_title("检测结果可视化", fontsize=SINGLE_DISPLAY_CONFIG["font_size"], pad=10)

    # 2. 绘制右侧文字结果
    result_text = create_result_text(vehicle_count, vehicle_detail, img_name)
    ax_text.text(
        0.05, 0.95, result_text,
        transform=ax_text.transAxes,
        fontsize=SINGLE_DISPLAY_CONFIG["font_size"],
        verticalalignment='top',
        horizontalalignment='left',
        fontweight='normal',
        family='SimHei',
        color='#2c3e50'
    )
    ax_text.axis('off')  # 关闭文本区域坐标轴
    ax_text.set_facecolor('#f8f9fa')  # 浅灰色背景提升可读性
    ax_text.set_title("检测统计信息", fontsize=SINGLE_DISPLAY_CONFIG["font_size"], pad=10)

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # 显示窗口（阻塞模式，关闭后继续）
    plt.show(block=True)

    # 关闭画布释放资源
    plt.close(fig)


def batch_detect_vehicles():
    """批量检测目录下所有车辆图片（单张独立弹窗展示）"""
    # 1. 获取所有有效图片路径
    img_paths = get_all_car_images()

    # 初始化汇总统计
    total_processed = 0
    total_success = 0
    total_vehicles_all = 0
    type_total_all = {v_name: 0 for v_name in VEHICLE_LABELS.values()}

    # 2. 逐张检测并展示
    print("\n🔍 开始逐张检测密集车辆（YOLOv8x）：")
    print("-" * 80)

    for i, img_path in enumerate(img_paths, 1):
        img_name = os.path.basename(img_path)
        total_processed += 1
        print(f"\n[{i}/{len(img_paths)}] 处理图片：{img_name}")

        # 单张图片检测
        results, vehicle_count, vehicle_detail, _, raw_img = detect_vehicles_single(img_path)

        if results is None:
            print(f"  ❌ 检测失败，跳过")
            continue

        total_success += 1
        total_vehicles_all += vehicle_count
        # 更新类别汇总
        for v_type, v_count in vehicle_detail.items():
            type_total_all[v_type] += v_count

        # 打印当前图片检测结果
        print(f"  ✅ 检测完成 | 总车辆数：{vehicle_count}")
        for v_type, v_count in vehicle_detail.items():
            if v_count > 0:
                print(f"    - {v_type}: {v_count} 辆")

        # 独立弹窗展示当前图片结果
        print(f"  🖼️ 显示检测结果窗口（关闭窗口继续下一张）...")
        display_single_image_result(results, vehicle_count, vehicle_detail, img_name, raw_img)

    # 3. 最终汇总统计
    print("\n" + "=" * 80)
    print("📊 批量检测最终汇总（密集车辆场景）：")
    print("=" * 80)
    print(f"总处理图片数：{total_processed}")
    print(f"成功检测图片数：{total_success}")
    print(f"检测失败图片数：{total_processed - total_success}")
    print(f"\n累计检测到车辆总数：{total_vehicles_all} 辆")
    print("\n累计车辆类别分布：")
    for v_type, count in type_total_all.items():
        if count > 0:
            ratio = count / total_vehicles_all * 100 if total_vehicles_all > 0 else 0
            print(f"  • {v_type}: {count} 辆（{ratio:.1f}%）")
    print("=" * 80)


# ===================== 程序运行入口 =====================
if __name__ == "__main__":
    try:
        batch_detect_vehicles()
        print("\n🎉 所有图片检测完成！")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        # 确保所有Matplotlib窗口关闭
        plt.close('all')