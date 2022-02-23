import os
import cv2
import numpy as np

# 在背景的某个位置插入文本
class ShxBackgroundRender:
    def __init__(self, img, font="hztxt.shx"):
        '''
        初始化
        :param img: 背景图片
        :param font: 字体文件路径
        '''
        self.img = img
        self.font = font
        self.origin = [0, 0]

    def __call__(self, s, x=0, y=100, sf=1.0):
        '''
        在坐标（x，y）位置渲染字符串s，缩放比例为sf
        :param s: 待渲染的字符串
        :param x: 横坐标（向右增长）
        :param y: 纵坐标（向下增长）
        :param sf: 缩放比例
        :return: 渲染后的背景图片
        '''
        if isinstance(s, str):
            self.sf = sf
            self.origin = [x, y]
            self.w = int(125 * self.sf)
            self.pad = int(50 * self.sf)
            box = self.render_string(s)
        elif isinstance(s, list):
            self.render_strings(s)
            box = None
        return self.img, box

    # 渲染若干字符串
    def render_strings(self, strs):
        '''
        参考代码，渲染一系列文本，每个文本的坐标不同，缩放比例也可以不同
        :param strs: 格式[str, [int, int], float]，表示文本、坐标和缩放比例，例如["测试", [100, 200], 0.5]
        :return: 返回渲染后的背景图
        '''
        text = ''
        for s in strs:          # 把所有待渲染的文本组成一个字符串，一次性交给shxpatch解析
            text += s[0]
        if len(text) > 0:
            lines = os.popen('shxpatch.exe' + ' simplex.shx ' + self.font + ' ' + text).readlines()
            cur = 0             # 表示当前处理的这一段文本在lines中的起始位置
            for s in strs:      # 对每段文本，配置原点和大小，然后渲染
                next = cur + len(s[0])
                self.origin = s[1]
                self.sf = s[2]
                self.w = int(125 * self.sf)
                for line in lines[cur:next]:
                    self.render_char(line, s[3])
                    self.origin[0] += self.w
                cur = next


    # 渲染单个字符串
    def render_string(self, string):
        box = None
        if len(string) > 0:
            lines = os.popen('shxpatch.exe' + ' simplex.shx ' + self.font + ' ' + string).readlines()
            box = self.render_lines(string, lines)
        return box

    def render_lines(self, string, lines):
        box = None
        if len(string) != len(lines):
            print('error: len(string) != len(lines)')
            return box
        for c, line in zip(string, lines):
            if '\u4e00' <= c <= '\u9fff':  # 汉字
                c_box = self.render_char(line)
            elif c in "acegmnopqrsuvwxyz":
                c_box = self.render_char(line, offset=[10, -15], sf=5.5)
            else:
                c_box = self.render_char(line, offset=[10, -22], sf=5.5)
            if box is None:
                box = c_box
                print(c_box)
            else:
                if box[0] > c_box[0]:
                    box[0] = c_box[0]
                if box[1] > c_box[1]:
                    box[1] = c_box[1]
                if box[2] < c_box[2]:
                    box[2] = c_box[2]
                if box[3] < c_box[3]:
                    box[3] = c_box[3]
            # self.origin[0] += self.w      # 固定字宽
            self.origin[0] = box[2] + self.pad  # 固定间距
        return box


    # 渲染单个字符
    def render_char(self, s, color=(0, 0, 0), thick=2, offset=None, sf=1.0):
        '''
        :param s:   待处理的字的字形信息
        :param color: 颜色
        :param thick: 粗细
        :param offset: 相对原点的偏移
        :param sf: 相对缩放比例
        :return: 包围盒左上角和右下角：[xmin, ymin, xmax, ymax]
        '''
        if offset is None:
            offset = [0, 0]
        sf = sf * self.sf
        canvas = self.img
        origin = [int(x * sf + y) for x, y in zip(offset, self.origin)]
        cur = origin
        box = None
        for op in s.split('|'):
            ops = op.split(':')
            if ops[0] == "moveto":
                cur = [float(x) * sf for x in ops[1].split(',')]
                cur = [int(x + y) for x, y in zip(cur, origin)]
            elif ops[0] == "lineto":
                to = [float(x) * sf for x in ops[1].split(',')]
                to = [int(x + y) for x, y in zip(to, origin)]
                cv2.line(canvas, cur, to, color, thick)
                if box is None:     # 初始化box
                    box = [cur[0], cur[1], cur[0], cur[1]]  # minx, miny, maxx, maxy
                    print(box)
                cur = to
                if cur[0] < box[0]:
                    box[0] = cur[0]
                if cur[1] < box[1]:
                    box[1] = cur[1]     # bug: cur[0]
                if cur[0] > box[2]:
                    box[2] = cur[0]
                if cur[1] > box[3]:
                    box[3] = cur[1]
            elif ops[0] == "anglearc":
                raise RuntimeError("anglearc not implement")
        return box


if __name__ == '__main__':
    # img = cv2.imread('test.jpg')
    img = np.zeros((600, 2000, 3), dtype="uint8")
    img.fill(255)
    rd = ShxBackgroundRender(img)
    # rd1 = ShxBackgroundRender(img, font="simplex.shx")
    # rd2 = ShxBackgroundRender(img, font="gbcbig.shx")
    cv2.circle(img, (100, 300), 3, (127, 127, 127), 1)               # 红圈是“测试”文本的原点，可见是左下角
    # rd([["字", [300, 300], 1, (255, 0, 0)]])
    # rd2([["字", [300, 300], 1, (0, 255, 0)]])
    # rd1([["A", [300, 300], 1, (0, 0, 255)]])

    # rd("FM甲0921", 100, 300, 1.0)
    # rd("FM甲0921", 100, 400, 0.5)
    _, box = rd("甲乙丙abcdefghijklmnopqrstuvwxyz", 100, 500, 1)
    # _, box = rd("甲", 100, 500, 0.5)
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 0))
    # rd.img = cv2.GaussianBlur(img, (5, 5), 0)
    # print(box)

    # rd([["你好", [200, 300], 0.2], ["三零三", [300, 400], 0.5]])   # 一次渲染多段文本
    # rd([["FM", [200, 300], 1], ["ABC", [300, 400], 1]])
    # rd([["I一I", [200, 400], 1]])
    # rd("测试", 100, 200, sf=0.8)
    cv2.imwrite('test2.jpg', rd.img)
