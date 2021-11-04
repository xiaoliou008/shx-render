# shx render

## 说明
可以渲染CAD的shx格式字体，同时也支持常见的ttf格式字体  

## 使用方式
使用render.py中的gen_data函数
```python
gen_data('测试test')
```
可以得到opencv格式的图片

保存到本地
```python
cv2.imwrite('test.jpg', gen_data("测试test"))
```