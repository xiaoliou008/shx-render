import os
import shutil
import time
import re
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2
from tqdm import tqdm
from cv2util import debug_image, cn_imwrite, cn_imread


# 使用shxpatch.exe解析shx文件，然后渲染文字
class ShxRender:
    def __init__(self, height=60, fontpath="gbcbig.shx"):
        self.sf = height / 230
        # self.sf = 1     # debug for hztxt
        self.h = height
        self.w = int(125 * self.sf)
        # self.w = int(500 * self.sf)     # debug
        self.dic = {}
        self.fontpath = fontpath

    def __call__(self, s: str):
        return self.render(s)

    def render(self, s: str):
        pad = int(5 * self.sf)
        img = np.zeros((self.h, pad), dtype="uint8")
        self.render_string(''.join([c for c in s if not c in self.dic]))
        for c in s:
            img = np.hstack((img, self.dic[c][:, pad:]))
        return cv2.bitwise_not(img)

    def render_string(self, string):
        if len(string) > 0:
            lines = os.popen('old_version\\shxpatch_202022.exe' + ' ' + self.fontpath + ' ' + string).readlines()
            for c, line in zip(string, lines):
                self.dic[c] = self.render_char(line)

    def render_char(self, s, thick=1):
        canvas = np.zeros((self.h, self.w), dtype="uint8")
        cur = origin = [0, 200 * self.sf]
        for op in s.split('|'):
            ops = op.split(':')
            if ops[0] == "moveto":
                cur = [float(x) * self.sf for x in ops[1].split(',')]
                cur = [int(x + y) for x, y in zip(cur, origin)]
            elif ops[0] == "lineto":
                to = [float(x) * self.sf for x in ops[1].split(',')]
                to = [int(x + y) for x, y in zip(to, origin)]
                cv2.line(canvas, cur, to, 255, thick)
                cur = to
            elif ops[0] == "anglearc":
                raise RuntimeError("anglearc not implement")
        return canvas

class Generator:
    def __init__(self):
        self.rd = ShxRender()

    def __call__(self, train_path=None, test_path=None, index=''):
        if train_path is not None and len(train_path) > 0:
            self.gen_train(train_path, append=False, idx=index)
        if test_path is not None and len(test_path) > 0:
            self.gen_test(test_path, append=True, idx=index)

    def gen_train(self, data_path, save_path="./train_data", append=True, idx=''):
        if not append:
            shutil.rmtree(save_path)
        train_path, _ = self.mkdir(idx=idx)
        train_fp = open(os.path.join(save_path, "rec", "train_list" + idx + ".txt"), 'a+' if append else 'w+', encoding="utf-8")
        with open(data_path, 'r', encoding="utf-8") as fp:
            lines = fp.readlines()
            for line in tqdm(lines):
                name = line.strip()
                img = self.gen_img(name)
                fname = name.replace('<', "&lt;").replace('>', "&gt;")
                cn_imwrite(os.path.join(train_path, fname + '.jpg') , img)
                name_path = "rec/train" + idx + "/" + fname + ".jpg"
                train_fp.write(name_path + '\t' + name + '\n')

    def gen_test(self, data_path, save_path="./train_data", append=True, idx=''):
        if not append:
            shutil.rmtree(save_path)
        _, test_path = self.mkdir(idx=idx)
        train_fp = open(os.path.join(save_path, "rec", "val_list" + idx + ".txt"), 'a+' if append else 'w+', encoding="utf-8")
        with open(data_path, 'r', encoding="utf-8") as fp:
            lines = fp.readlines()
            for line in tqdm(lines):
                name = line.strip()
                img = self.gen_img(name)
                fname = name.replace('<', "&lt;").replace('>', "&gt;")
                cn_imwrite(os.path.join(test_path, fname + '.jpg') , img)
                name_path = "rec/test" + idx + "/" + fname + ".jpg"
                train_fp.write(name_path + '\t' + name + '\n')

    def mkdir(self, save_path="./train_data", idx=''):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if not os.path.exists(os.path.join(save_path, "rec")):
            os.mkdir(os.path.join(save_path, "rec"))
        train_path = os.path.join(save_path, "rec", "train" + idx)
        if not os.path.exists(train_path):
            os.mkdir(train_path)
        test_path = os.path.join(save_path, "rec", "test" + idx)
        if not os.path.exists(test_path):
            os.mkdir(test_path)
        return (train_path, test_path)

    def gen_img(self, s: str, font_size=50, height=60):
        words = re.findall(r'[\u4e00-\u9fff]+|[^\u4e00-\u9fff]+', s)
        canvas = np.zeros((height, 1), dtype="uint8")
        canvas[:] = 255
        for word in words:
            if re.match(r'[\u4e00-\u9fff]+', word):  # 中文用gbcbig.shx字体
                img = self.rd(word)
                # img = cv2.resize(img, dsize=None, fx=height / 230, fy=height / 230, interpolation=cv2.INTER_AREA)
                # _, img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)
            else:  # 英文用Simplex字体
                num = len(word)
                width = font_size * num
                image = Image.new("RGB", (width, height), (255, 255, 255))
                draw = ImageDraw.Draw(image)
                font = ImageFont.truetype('Simplex.ttf', font_size, encoding='utf-8')
                draw.text((0, 10), word, (0, 0, 0), font=font)
                image = image.resize((int(width * 0.8), height), Image.ANTIALIAS)
                img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 获取灰度图
                (thresh, img_bin) = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)  # 二值化
                img = img_bin[:, : draw.textsize(word, font)[0]]
                for i in range(len(img[0]) - 1, -1, -1):
                    if sum(img[:, i]) < 255 * len(img):
                        img = img[:, :i + 5]
                        break
            # canvas = np.hstack((canvas, np.zeros((height, 1), dtype="uint8"), img))
            canvas = np.hstack((canvas, img))
        pad = np.zeros((height, 10), dtype="uint8")
        pad[:] = 255
        canvas = cv2.resize(np.hstack((canvas, pad)), dsize=None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
        # _, canvas = cv2.threshold(canvas, 250, 255, cv2.THRESH_BINARY)
        return canvas


if __name__ == '__main__':
    # start = time.time()
    # rd = ShxRender(height=600, fontpath="hztxt.shx")
    # rd = ShxRender(fontpath="gbcbig.shx")
    # rd2 = ShxRender(height=600, fontpath="simplex.shx")
    # rd = ShxRender(height=600, fontpath="_HZTXT.SHX")
    # cv2.imwrite('test2.jpg', rd("洞口尺寸"))
    gen = Generator()
    # gen("data.txt", "")

    # gen("text/data2.txt", "text/test2.txt", index='2')
    # gen("text/data3.txt", "text/test3.txt", index='3')
    gen("text/data4.txt", "text/test4.txt", index='4')

    # cv2.imwrite('test.jpg', gen.gen_img("FM乙0921"))
    # cv2.imwrite('test.jpg', gen.gen_img("CABC2421a"))
    # cv2.imwrite('test.jpg', gen.gen_img("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    # cv2.imwrite('test.jpg', gen.gen_img("门槛"))
    # cv2.imwrite('test.jpg', gen.gen_img("G2421a"))
    # print('运行时间：{} s'.format(time.time() - start))
