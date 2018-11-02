import os

ALLOWED_EXTENSIONS = set(['png', 'PNG', 'jpg', 'JPG', 'gif', 'GIF'])


# 用于判断文件后缀
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)
    os.rmdir(path)


if __name__ == '__main__':
    f = os.path.join('E://ocr//upload//image', 't1')
    del_file(f)
