import os
import random

from cv2 import imwrite
from tqdm import tqdm, trange
from PIL import Image, ImageFont, ImageDraw
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

import cv2
import numpy as np
# from renderText import ShxBackgroundRender
from math import *
import copy
import json
from shapely.geometry import Polygon as geo_Polygon


l1 = ['甲', '乙', '丙', '丁', '戊']
l2 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
      'x', 'y', 'z']
l3 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
      'X', 'Y', 'Z']
l4 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']


char_dict = {}

# 读取图片
folder = 'jpg'
new_folder = 'new_jpg'
sub_folder = 'split_images'

step_w = 500
step_h = 500
sub_w = 1024
sub_h = 1024


# 门窗编号格式
# M[0-9] FM甲/乙/丙[0-9] LC/GC/LM/DM[0-9][0-9][a-z][0-9]（[a-z]）  带连接符的
# def generate_door_num():
#     seq = ''
#     n1 = random.randint(0, 4)
#     if n1 == 0:
#         n = random.randint(0, 9)
#         return 'M' + str(n)
#     elif n1 == 1:
#         seq += 'FM'
#     elif n1 == 2:
#         seq += 'TLM'
#     elif n1 == 3:
#         seq += 'GBM'
#     elif n1 == 4:
#         seq += 'LM'
#     n = random.randint(0, 3)
#     seq += l1[n]
#     n = random.randint(0, 9)
#     seq += str(n)
#     n2 = random.randint(0, 3)
#     for i in range(n2):
#         n3 = random.randint(0, 35)
#         if n3 < 10:
#             seq += str(n3)
#         else:
#             seq += l2[n3 - 10]
#     return seq
#
#
# def generate_window_num():
#     seq = ''
#     n1 = random.randint(0, 2)
#     if n1 == 0:
#         n = random.randint(0, 9)
#         return 'c' + str(n)
#     elif n1 == 1:
#         seq += 'BYC'
#     elif n1 == 2:
#         seq += 'LC'
#     n = random.randint(0, 9)
#     seq += str(n)
#     n2 = random.randint(0, 3)
#     for i in range(n2):
#         n3 = random.randint(0, 35)
#         if n3 < 10:
#             seq += str(n3)
#         else:
#             seq += l2[n3 - 10]
#     return seq

# 生成num个小写字母
def gen_random_low(num):
    return ''.join(random.sample("zyxwvutsrqponmlkjihgfedcba", num))

# 生成num个大写字母
def gen_random_upper(num):
    return gen_random_low(num).upper()

# 生成纯数字
def gen_random_numstr(num):
    return ''.join(random.sample("1234567890", num))

# 生成含有减号-的数字序列
def gen_random_numnew(num):
    return ''.join(random.sample("1234567890-", num))

# 生成文本序列（门窗编号类）
def generate_num_seq():
    n = random.randint(0, 9)
    if n == 9:
        if random.randint(0, 1) == 0:
            return 'M' + random.sample('1234567890', 1)[0]
        else:
            return 'C' + random.sample('1234567890', 1)[0]
    hx = random.randint(0, 4)
    line = gen_random_upper(random.randint(1, 3))
    if random.randint(0, 1) == 0:
        line = line + random.sample('甲乙丙丁戊', 1)[0]
    indexl = len(line)
    line = line + gen_random_numstr(random.randint(2, 4))
    if random.randint(0, 1) == 0:
        line = line + gen_random_low(1)
    indexr = len(line)
    if hx == 0:
        index = random.randint(indexl, indexr - 1)
        return line[:index] + '-' + line[index:]
    return line

# 生成文本标签（25%概率生成房间名，75%门窗编号）
def generate_word():
    type = random.randint(0, 4)
    if type == 0:
        if random.randint(0, 8) == 0:
            word = '卫' + gen_random_numstr(1)
            if random.randint(0, 5) == 0:
                word += 'a'
        else:
            index = random.randint(0, len(names) - 1)
            word = names[index]
    else:
        word = generate_num_seq()
    return word

# 生成一个随机的角度
def generate_angle():
    an = random.randint(0, 9)
    if an <= 3:
        return 0
    elif an <= 5:
        return -90
    elif an <= 7:
        return 90
    elif an == 8:
        m = random.randint(0, 3)
        angle_list = [-60, -30, 30, 60]
        return angle_list[m]
    else:
        m = random.randint(0, 3)
        angle_list = [-53, -37, 37, 53]
        return angle_list[m]

# 旋转图片
def rotate(image, degree, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    # 计算旋转之后的图片大小
    heightNew = int(w * fabs(sin(radians(degree))) + h * fabs(cos(radians(degree))))
    widthNew = int(h * fabs(sin(radians(degree))) + w * fabs(cos(radians(degree))))
    M = cv2.getRotationMatrix2D(center, degree, scale)
    M[0, 2] += (widthNew - w) / 2
    M[1, 2] += (heightNew - h) / 2
    rotated = cv2.warpAffine(image, M, (widthNew, heightNew), borderValue=(255, 255, 255))
    # 背景设置成白色
    return rotated



# 渲染单个字符
def render_char(s, img, origin, color=(0, 0, 0), thick=2, offset=None, sf=1.0, hw=1.0):
    '''
    :param s:   待处理的字的字形信息
    :param color: 颜色
    :param thick: 粗细
    :param offset: 相对原点的偏移
    :param sf: 相对缩放比例
    :param hw: 用于调整长宽比
    :return: 包围盒左上角和右下角：[xmin, ymin, xmax, ymax]
    '''
    if offset is None:
        offset = [0, 0]
    canvas = img
    origin = [int(x * sf + y) for x, y in zip(offset, origin)]
    cur = origin
    box = [cur[0], cur[1], cur[0], cur[1]]  # minx, miny, maxx, maxy
    for op in s.split('|'):
        ops = op.split(':')
        if ops[0] == "moveto":
            cur = [float(x) * sf for x in ops[1].split(',')]
            cur[0] *= hw
            cur = [int(x + y) for x, y in zip(cur, origin)]

        elif ops[0] == "lineto":
            to = [float(x) * sf for x in ops[1].split(',')]
            to[0] *= hw
            to = [int(x + y) for x, y in zip(to, origin)]

            cv2.line(canvas, cur, to, color, thick)
            cur = to
        elif ops[0] == "anglearc":
            raise RuntimeError("anglearc not implement")
        if cur[0] < box[0]:
            box[0] = cur[0]
        if cur[1] < box[1]:
            box[1] = cur[1]
        if cur[0] > box[2]:
            box[2] = cur[0]
        if cur[1] > box[3]:
            box[3] = cur[1]
    return canvas, box


# 渲染单个字符串
def render_lines(string, origin, img):
    box = None

    gaussian_ratio = random.normalvariate(1, 0.25)
    if gaussian_ratio < 0.8:
        gaussian_ratio = 0.8
    elif gaussian_ratio > 1.25:
        gaussian_ratio = 1.25

    thick = 2
    a = random.randint(0, 4)
    if a == 3:
        thick = 1
    elif a == 4:
        thick = 3

    h_w_radio = random.uniform(0.6, 1.0)

    for c in string:
        if '\u4e00' <= c <= '\u9fff':  # 汉字
            img, c_box = render_char(char_dict[c], img, origin, thick=thick, sf=0.45 * gaussian_ratio, hw=h_w_radio)
        elif c in "acegmnopqrsuvwxyz":
            img, c_box = render_char(char_dict[c], img, origin, thick=thick, offset=[8, -15], sf=2.475 * gaussian_ratio, hw=h_w_radio)
        elif c == '-':
            img, c_box = render_char(char_dict[c], img, origin, thick=thick, offset=[8, -13], sf=2.475 * gaussian_ratio, hw=h_w_radio)
        else:
            img, c_box = render_char(char_dict[c], img, origin, thick=thick, offset=[8, -22], sf=2.475 * gaussian_ratio, hw=h_w_radio)
        if box is None:
            box = c_box
        else:
            if box[0] > c_box[0]:
                box[0] = c_box[0]
            if box[1] > c_box[1]:
                box[1] = c_box[1]
            if box[2] < c_box[2]:
                box[2] = c_box[2]
            if box[3] < c_box[3]:
                box[3] = c_box[3]
        # 暂时固定间距
        pad = 2
        origin[0] = box[2] + pad
    return img, box


# 生成带文字的旋转后的图片
def get_word_img(word, angle):
    # 初始化底图
    canvas = np.zeros((180, 1000, 3), np.uint8)
    canvas.fill(255)
    img, box = render_lines(word, [5, 90], canvas)
    wt = 5
    img = img[max(box[1] - wt, 0):box[3] + wt, max(box[0] - wt, 0):box[2] + wt]
    h1, w1, _ = img.shape
    img_ro = rotate(img, angle)
    h2, w2, _ = img_ro.shape
    box = []
    # box表示左上、右上、右下、左下四个点的x,y坐标
    if angle >= 0:
        box.append(0)
        box.append(int(w1 * sin(radians(angle))))
        box.append(w2 - int(h1 * sin(radians(angle))))
        box.append(0)
        box.append(w2)
        box.append(int(h1 * cos(radians(angle))))
        box.append(int(h1 * sin(radians(angle))))
        box.append(h2)
    else:
        an = fabs(angle)
        box.append(int(h1 * sin(radians(an))))
        box.append(0)
        box.append(w2)
        box.append(int(w1 * sin(radians(an))))
        box.append(w2 - int(h1 * sin(radians(an))))
        box.append(h2)
        box.append(0)
        box.append(int(h1 * cos(radians(an))))

    # cv2.line(img_ro, (box[0], box[1]), (box[2], box[3]), (0, 0, 255))
    # cv2.line(img_ro, (box[2], box[3]), (box[4], box[5]), (255, 0, 0))
    # cv2.line(img_ro, (box[4], box[5]), (box[6], box[7]), (255, 0, 0))
    # cv2.line(img_ro, (box[6], box[7]), (box[0], box[1]), (255, 0, 0))
    # cv2.imwrite('tmp.jpg', img_ro)
    return img_ro, box


def bbox_IOU(bbox_a, bbox_b):
    a_xmin, a_ymin, a_xmax, a_ymax = bbox_a[0], bbox_a[1], bbox_a[2], bbox_a[3]
    b_xmin, b_ymin, b_xmax, b_ymax = bbox_b[0], bbox_b[1], bbox_b[2], bbox_b[3]
    if b_xmin > a_xmax or a_xmin > b_xmax or b_ymin > a_ymax or a_ymin > b_ymax:
        return 0
    a_area = (a_xmax - a_xmin) * (a_ymax - a_ymin)
    b_area = (b_xmax - b_xmin) * (b_ymax - b_ymin)
    x_min = max(a_xmin, b_xmin)
    x_max = min(a_xmax, b_xmax)
    y_min = max(a_ymin, b_ymin)
    y_max = min(a_ymax, b_ymax)
    overlap_area = (x_max - x_min) * (y_max - y_min)
    if overlap_area <= 0:
        return 0
    return overlap_area / (a_area + b_area - overlap_area)


def cal_overlap(bbox_a, bbox_b):

    element_a_x_min = min(bbox_a[0], bbox_a[2], bbox_a[4], bbox_a[6])
    element_a_x_max = max(bbox_a[0], bbox_a[2], bbox_a[4], bbox_a[6])
    element_a_y_min = min(bbox_a[1], bbox_a[3], bbox_a[5], bbox_a[7])
    element_a_y_max = max(bbox_a[1], bbox_a[3], bbox_a[5], bbox_a[7])

    element_b_x_min = min(bbox_b[0], bbox_b[2], bbox_b[4], bbox_b[6])
    element_b_x_max = max(bbox_b[0], bbox_b[2], bbox_b[4], bbox_b[6])
    element_b_y_min = min(bbox_b[1], bbox_b[3], bbox_b[5], bbox_b[7])
    element_b_y_max = max(bbox_b[1], bbox_b[3], bbox_b[5], bbox_b[7])

    if element_b_x_min > element_a_x_max or element_a_x_min > element_b_x_max or element_b_y_min > element_a_y_max or element_a_y_min > element_b_y_max:
        return 0


    geo_polygon_a = np.asarray(bbox_a)
    geo_polygon_b = np.asarray(bbox_b)
    geo_polygon_a = geo_Polygon(geo_polygon_a[:8].reshape((4,2)))
    geo_polygon_b = geo_Polygon(geo_polygon_b[:8].reshape((4,2)))
    if not geo_polygon_a.is_valid or not geo_polygon_b.is_valid:
        return 0

    a_area = geo_polygon_a.area
    b_area = geo_polygon_b.area
    overlap_area = geo_polygon_a.intersection(geo_polygon_b).area
    if overlap_area < 0:
        print('ERROR:0-MINUS OVERLAP AREA')
        return 0
    if min(a_area, b_area)==0:
        return 0
    return overlap_area


# 加载空间名称词库
f = open('room_set.txt')
names = f.readlines()
for i in range(len(names)):
    names[i] = names[i][:-1]
f.close()


# 构建所有需要解析的字符
gl_str = '-'
for name in names:
    for c in name:
        if c not in char_dict:
            char_dict[c] = ''
            gl_str += c
for c in l1 + l2 + l3 + l4:
    if c not in char_dict:
        char_dict[c] = ''
        gl_str += c

# 把所有字的笔画都预存到内存中
lines = os.popen('shxpatch.exe' + ' simplex.shx ' + 'hztxt.shx ' + gl_str).readlines()
for c, line in zip(gl_str, lines):
    char_dict[c] = line



total_num = 0

def generate_subimg():
    # 先生成所有1024*1024非全白的子图
    for _, _, files in os.walk(folder):
        for jpg_name in tqdm(files):
            inn = 0
            jpg_path = os.path.join(folder, jpg_name)
            # opencv读入图片
            img_total = cv2.imdecode(np.fromfile(jpg_path, dtype=np.uint8), -1)

            w, h = img_total.shape[1], img_total.shape[0]

            for x in range(0, w, step_w):
                for y in range(0, h, step_h):

                    # 裁剪出子图
                    h_ = min(sub_h, h - y)
                    w_ = min(sub_w, w - x)
                    if h_ == 1024 and w_ == 1024:
                        # sub_img：原图的对应子图  用来统计黑色像素占比 --- 是否要在其上生成数据
                        sub_img = img_total[y: y + h_, x: x + w_]
                        if np.count_nonzero(sub_img) < (h_ * w_ - 1000) * 3:
                            cv2.imencode('.jpg', sub_img)[1].tofile('split_images\\' + jpg_name[:-4] + '_' + str(inn) + '.jpg')
                            inn += 1

# 生成子图
generate_subimg()

# 开始处理图片
for _, _, files in os.walk(sub_folder):
    for jpg_name in tqdm(files):
        wordlist = []
        boxlist = []
        anglelist = []

        jpg_path = os.path.join(sub_folder, jpg_name)

        img_total = cv2.imdecode(np.fromfile(jpg_path, dtype=np.uint8), -1)

        # 确定文本标签数量
        text_tmpnum = random.randint(5, 10)     # 每张小图里有5~10个标签
        text_num = 0

        for i in range(text_tmpnum):
            # 确定文本图片及旋转小图
            word = generate_word()
            angle = generate_angle()
            word_img, box = get_word_img(word, angle)
            h2, w2, _ = word_img.shape
            flag = 0
            try_num = 3
            # 保证生成的标签不会重叠，当尝试了3次都可能重叠后，就放弃
            while flag == 0 and try_num > 0:
                # 随机生成贴的位置
                tx = random.randint(0, sub_w - w2)
                ty = random.randint(0, sub_h - h2)
                new_box = copy.deepcopy(box)
                for j in range(8):
                    if j % 2 == 0:
                        new_box[j] += tx
                    else:
                        new_box[j] += ty

                # 先计算交并比（新的box和已经贴好的box）
                for itemBox in boxlist:
                    if cal_overlap(new_box, itemBox) > 0:
                        flag = 1
                        break
                if flag == 0:
                    tmp = img_total[ty: ty + h2, tx: tx + w2]
                    tmp = cv2.bitwise_and(tmp, word_img)
                    img_total[ty: ty + h2, tx: tx + w2] = tmp
                    wordlist.append(word)
                    boxlist.append(new_box)
                    anglelist.append(angle)
                    text_num += 1
                    flag = 1
                else:
                    flag = 0
                    try_num -= 1


        print("begin to save img and json")

        cv2.imencode('.jpg', img_total)[1].tofile('new_jpg\\' + jpg_name)

        for box in boxlist:
            cv2.line(img_total, (box[0], box[1]), (box[2], box[3]), (0, 0, 255))
            cv2.line(img_total, (box[2], box[3]), (box[4], box[5]), (255, 0, 0))
            cv2.line(img_total, (box[4], box[5]), (box[6], box[7]), (255, 0, 0))
            cv2.line(img_total, (box[6], box[7]), (box[0], box[1]), (255, 0, 0))

        cv2.imencode('.jpg', img_total)[1].tofile('label_img\\' + jpg_name)

        text_json = {}
        text_json['jpg_path'] = 'new_jpg\\' + jpg_name
        text_num = len(wordlist)
        text_json['text_num'] = text_num
        text_json['text_label'] = []
        for i in range(text_num):
            tmp_dict = {}
            tmp_dict['text'] = wordlist[i]
            tmp_dict['position'] = {
                'x0': boxlist[i][0],
                'y0': boxlist[i][1],
                'x1': boxlist[i][2],
                'y1': boxlist[i][3],
                'x2': boxlist[i][4],
                'y2': boxlist[i][5],
                'x3': boxlist[i][6],
                'y3': boxlist[i][7],
            }
            tmp_dict['angle'] = anglelist[i]
            text_json['text_label'].append(tmp_dict)

        with open('label_text\\' + jpg_name[:-4] + '.json', "w+", encoding="UTF-8") as f:
            json.dump(text_json, f, ensure_ascii=False)
        total_num += text_num

print(total_num)
