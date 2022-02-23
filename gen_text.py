import os
import random

archi = None


class ArchiWords:
    def __init__(self, path="建筑领域/archi-word"):
        self.words = []
        for file_name in os.listdir(path):
            fpath = path + '/' + file_name
            with open(fpath, encoding='utf-8') as fp:
                for line in fp.readlines():
                    self.words.append(line.strip())

    def get_random_words(self, num):
        return ''.join(random.sample(self.words, num))


def gen_random_low(num):
    return ''.join(random.sample("zyxwvutsrqponmlkjihgfedcba", num))


def gen_random_upper(num):
    return gen_random_low(num).upper()


def gen_random_numstr(num):
    return ''.join(random.sample("1234567890", num))


def gen_save_text(path, num):
    with open(path, 'w+', encoding="utf-8") as fp:
        for i in range(num):
            line = gen_random_upper(random.randint(1, 3))
            if random.random() > 0.0:
                line = line + random.sample('甲乙乙丙丁戊', 1)[0]
            line = line + gen_random_numstr(random.randint(2, 4))
            line = line + random.sample('adgq', 1)[0]
            fp.write(line + '\n')


def gen_save_text2(path, num):
    with open(path, 'w+', encoding="utf-8") as fp:
        for i in range(int(num / 2)):
            line = gen_random_upper(random.randint(1, 3))
            line = line + gen_random_numstr(random.randint(2, 4)) + 'a'
            line = line + '(' + gen_random_upper(1) + gen_random_numstr(1) + '.' + gen_random_numstr(2)
            line = line + random.sample('甲甲乙乙丙丁戊', 1)[0] + '级)'
            fp.write(line + '\n')
        for i in range(int(num / 4)):
            line = '要求' + random.sample('>><', 1)[0]
            line = line + gen_random_numstr(1) + '.' + gen_random_numstr(1) + 'h'
            fp.write(line + '\n')
        for i in range(int(num / 4)):
            line = gen_random_numstr(2) + '0' + '厚度' + '3' + gen_random_numstr(2)
            fp.write(line + '\n')


def gen_save_text3(path, num):
    with open(path, 'w+', encoding="utf-8") as fp:
        for i in range(int(num / 2)):
            line = gen_random_numstr(1) + '00' + '厚' + archi.get_random_words(2)
            fp.write(line + '\n')
        for i in range(int(num / 2)):
            line = archi.get_random_words(2) + '厚度30'
            fp.write(line + '\n')

# 博智林图纸的门窗编号
def gen_save_text4(path, num):
    gen_set = set()
    random.seed(303)
    with open(path, 'w+', encoding="utf-8") as fp:
        for i in range(int(num / 10) * 2):
            line = ''.join(random.sample("CM", 1))
            if i % 7 == 0:
                line = 'L' + line
            elif i % 7 == 1:
                line += 'D'
            line += gen_random_numstr(1)
            if line not in gen_set:
                fp.write(line + '\n')
                gen_set.add(line)
        for i in range(int(num / 10) * 7):
            line = ''.join(random.sample("FFLLLLLGS", 1)) + ''.join(random.sample("CM", 1))
            line += gen_random_numstr(2)
            line += ''.join(random.sample("aabc-", 1))
            line += gen_random_numstr(1)
            if i % 5 == 0:
                line += 'a'
            if line not in gen_set:
                fp.write(line + '\n')
                gen_set.add(line)
        for i in range(int(num / 100) * 1):
            line = '卫' + gen_random_numstr(1)
            if i % 3 == 0:
                line += 'a'
            if line not in gen_set:
                fp.write(line + '\n')
                gen_set.add(line)

def main():
    # gen_save_text('text/data.txt', 10000)
    # gen_save_text('text/test.txt', 1000)
    # gen_save_text2('text/data2.txt', 10000)
    # gen_save_text2('text/test2.txt', 1000)
    # gen_save_text3('text/data3.txt', 1000)
    # gen_save_text3('text/test3.txt', 100)
    gen_save_text4('text/data4.txt', 10000)
    gen_save_text4('text/test4.txt', 1000)


if __name__ == '__main__':
    archi = ArchiWords()
    main()


# import random
#
# def gen_random_low(num):
#     return ''.join(random.sample("zyxwvutsrqponmlkjihgfedcba",num))
#
# def gen_random_upper(num):
#     return gen_random_low(num).upper()
#
# def gen_random_numstr(num):
#     return ''.join(random.sample("1234567890", num))
#
# def gen_save_text(path, num):
#     with open(path, 'w+', encoding="utf-8") as fp:
#         for i in range(num):
#             line = gen_random_upper(random.randint(1, 3))
#             if random.random() > 0.0:
#                 line = line + random.sample('甲乙乙丙丁戊', 1)[0]
#             line = line + gen_random_numstr(random.randint(2, 4))
#             line = line + random.sample('adgq', 1)[0]
#             fp.write(line + '\n')
#
# def main():
#     gen_save_text('data.txt', 100000)
#     gen_save_text('test.txt', 10000)
#
#
# if __name__ == '__main__':
#     main()