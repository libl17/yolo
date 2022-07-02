import os
if __name__ == '__main__':
    path = './data/images/'
    imgs = os.listdir(path)
    for img in imgs:
        img_mod = img.replace('.JPG','.jpg')
        os.rename(path + img, path + img_mod)
