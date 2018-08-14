from PIL import Image

try:
    img = Image.open(pic_path)
except IOError:
    print(pic_path)
try:
    img = np.array(img, dtype=np.float32)
except:
    print('corrupt img', pic_path)