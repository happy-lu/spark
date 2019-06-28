from PIL import Image
img = Image.open('a.ppm')
img.save('a.bmp')
img.show()