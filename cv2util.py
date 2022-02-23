import numpy as np
import cv2

# 调试
def debug_image(image, msg="image"):
    cv2.namedWindow(msg, 0)
    cv2.imshow(msg, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 读取中文路径图片
def cn_imread(cn_path: str, flags=-1):
    return cv2.imdecode(np.fromfile(cn_path, dtype=np.uint8), flags)

# 读取和写入中文路径图片
def cn_imwrite(save_path: str, img_save):
    suffix = save_path[save_path.index('.'):]
    cv2.imencode(suffix, img_save)[1].tofile(save_path)