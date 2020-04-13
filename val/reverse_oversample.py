import os
import cv2

cats = ['fighting', 'standing', 'walking']
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

def reverse_duplicate(name, images_to_duplicate, cat, file):
    for image in images_to_duplicate:
        old_path = cat + "\\" + file + "\\" + str(image) + ".jpg"
        new_path = cat + "\\" + file + "\\" + str(name+1) + ".jpg"

        frame = cv2.imread(old_path)
        cv2.imwrite(new_path, frame)

        name += 1

for label, cat in enumerate(cats):
    files = os.listdir(cat)
    for file in files:
        print(cat, file)
        images = os.listdir(cat + '\\' + file)
        images = [int(image.replace('.jpg', '')) for image in images if '.avi' not in image]
        images.sort(reverse=True)

        if len(images) == 0:
            continue
        else:
            name = images[0]

        length = len(images)

        if length == 16:
            images_to_duplicate = images[1:] + images[:-1][::-1] + images[1:] + images[:-1][::-1]
            # print(name, images_to_duplicate, len(images_to_duplicate) + length)
            assert (len(images_to_duplicate) + length > 64)
            reverse_duplicate(name, images_to_duplicate, cat, file)

        elif 32 > length > 16: # need to x3
            images_to_duplicate = images[1:] + images[:-1][::-1] + images[1:]
            # print(name, images_to_duplicate, len(images_to_duplicate) + length)
            assert(len(images_to_duplicate) + length > 64)
            reverse_duplicate(name, images_to_duplicate, cat, file)
        elif length == 32:
            images_to_duplicate = images[1:] + images[:-1][::-1]
            # print(name, images_to_duplicate, len(images_to_duplicate) + length)
            assert(len(images_to_duplicate) + length > 64)
            reverse_duplicate(name, images_to_duplicate, cat, file)
        elif 64 > len(images) > 32:
            images_to_duplicate = images[1:]
            # print(name, images_to_duplicate, len(images_to_duplicate) + length)
            assert (len(images_to_duplicate) + length > 64)
            reverse_duplicate(name, images_to_duplicate, cat, file)
        else:
            print(cat, file, "skipped")

