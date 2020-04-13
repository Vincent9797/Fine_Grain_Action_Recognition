import os
import cv2

cats = ['fighting', 'standing', 'walking']
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
train = open("train_64frames.txt", "w")

def generate_2_64_frames(cat, file, images, label):
    first_64 = images[:64]
    last_64 = images[-64:]

    out = cv2.VideoWriter(cat + '\\' + file + '\\first_64frames.avi', fourcc, 10.0, (224, 224))

    for image in first_64:
        img = cv2.imread(cat + '\\' + file + '\\' + str(image) + '.jpg')
        out.write(img)
    out.release()

    out = cv2.VideoWriter(cat + '\\' + file + '\\last_64frames.avi', fourcc, 10.0, (224, 224))

    for image in last_64:
        img = cv2.imread(cat + '\\' + file + '\\' + str(image) + '.jpg')
        out.write(img)
    out.release()

    train.write('train\\' + cat + '\\' + file + '\\first_64frames ' + str(label) + '\n')
    train.write('train\\' + cat + '\\' + file + '\\last_64frames ' + str(label) + '\n')

for label, cat in enumerate(cats):
    files = os.listdir(cat)
    for file in files:
        print(cat, file)
        images = os.listdir(cat + '\\' + file)
        images = [int(image.replace('.jpg', '')) for image in images if '.avi' not in image]
        images.sort()

        if len(images) > 64:
        # max_stride = 0
        # generate(max_stride, cat, file, images, label)
            generate_2_64_frames(cat, file, images, label)