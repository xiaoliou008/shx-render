import os
import time
import re
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2

class ShxRender:
    def __init__(self):
        self.h = 230
        self.w = 125
        self.dic = {}

    def __call__(self, s: str):
        return self.render(s)

    def render(self, s: str):
        pad = 5
        img = np.zeros((self.h, pad), dtype="uint8")
        self.render_string(''.join([c for c in s if not c in self.dic]))
        for c in s:
            img = np.hstack((img, self.dic[c][:, pad:]))
        return cv2.bitwise_not(img)

    def render_string(self, string):
        if len(string) > 0:
            lines = os.popen('shxpatch.exe' + ' ' + string).readlines()
            for c, line in zip(string, lines):
                self.dic[c] = self.render_char(line)

    def render_char(self, s):
        canvas = np.zeros((self.h, self.w), dtype="uint8")
        cur = origin = [0, 200]
        for op in s.split('|'):
            ops = op.split(':')
            if ops[0] == "moveto":
                cur = [float(x) for x in ops[1].split(',')]
                cur = [int(x + y) for x, y in zip(cur, origin)]
            elif ops[0] == "lineto":
                to = [float(x) for x in ops[1].split(',')]
                to = [int(x + y) for x, y in zip(to, origin)]
                cv2.line(canvas, cur, to, 255, 2)
                cur = to
            elif ops[0] == "anglearc":
                raise RuntimeError("anglearc not implement")
        return canvas

def gen_data(s: str, font_size=100, height=120):
    words = re.findall(r'[\u4e00-\u9fff]+|[^\u4e00-\u9fff]+', s)
    canvas = np.zeros((height, 1), dtype="uint8")
    canvas[:] = 255
    rd = ShxRender()
    for word in words:
        if re.match(r'[\u4e00-\u9fff]+', word):         # 中文用gbcbig.shx字体
            img = rd(word)
            img = cv2.resize(img, dsize=None, fx=height/230, fy=height/230, interpolation=cv2.INTER_AREA)
        else:                                           # 英文用Simplex字体
            num = len(word)
            width = font_size * num
            image = Image.new("RGB", (width, height), (255, 255, 255))
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype('Simplex.ttf', font_size, encoding='utf-8')
            draw.text((0, 20), word, (0, 0, 0), font=font)
            image = image.resize((int(width * 0.8), height), Image.ANTIALIAS)
            img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                            # 获取灰度图
            (thresh, img_bin) = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)    # 二值化
            img = img_bin[:, : draw.textsize(word, font)[0]]
            for i in range(len(img[0]) - 1, -1, -1):
                if sum(img[:, i]) < 255 * len(img):
                    img = img[:, :i + 5]
                    break
        # canvas = np.hstack((canvas, np.zeros((height, 1), dtype="uint8"), img))
        canvas = np.hstack((canvas, img))
    pad = np.zeros((height, 10), dtype="uint8")
    pad[:] = 255
    return np.hstack((canvas, pad))


def debug_image(image, msg="image"):
    cv2.namedWindow(msg, 0)
    cv2.imshow(msg, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start = time.time()
    # rd = ShxRender()
    # cv2.imwrite('test.jpg', rd("淦"))
    cv2.imwrite('test.jpg', gen_data("FM乙0921"))
    print('运行时间：{} s'.format(time.time() - start))