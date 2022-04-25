'''
This file is all function we need for data preprocessing
the function of the directory we can modify, so that it will be a customed function
'''

import numpy as np
import json
import base64
import shutil
import glob
import cv2
from math import cos, sin, pi, fabs, radians
import os
import glob
import json
import csv
import random
import Data.labelme2coco as labelme2coco
from labelme import utils
from PIL import Image
def gaussian_noise(img, mean=0, sigma=0.1):

    img = img / 255
    noise = np.random.normal(mean, sigma, img.shape)
    gaussian_out = img + noise
    gaussian_out = np.clip(gaussian_out, 0, 1)
    gaussian_out = np.uint8(gaussian_out*255)
    return gaussian_out

def move_files(root, destpath):

    """
    :the directory root: the source file path of certain file type like ".jpg", ".json"
    and if you want to move all of the type of the files, just use it with "glob.glob("../*.json")"
    """
    for name in glob.glob(root):
        shutil.move(name, destpath)

def move_label(base_path, destpath, counter):
    """
    In this function, if your image name is a sequence like 1.png, 2.png, ..., 200.png,
    the computer will load your image like that: 1.png, 10.png, 11.png,..., 100.png,...,199.png, 2.png,
    it will cause your code or training image label confused and will also cause the correctness of weight be wrong
    so there are some recommend for you

    1000.png, 1001.png, .etc like this
    """

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == "label.png":

                print(root)
                shutil.move(root + "/" + file, destpath)

                newname = destpath + str(counter) + ".png"
                os.rename(destpath + r"label.png", newname)

                counter+=1

def rename(counter, correct):

    newname = r"C:\3D/Training_Image_1920_1728/cv2_mask/" + str(correct) + ".png"
    os.rename(r"C:\3D/Training_Image_1920_1728/cv2_mask/"+ str(counter) + ".png", newname)

def image_resize(filepath, destpath, size):

    if not os.path.exists(destpath):
        os.makedirs(destpath)

    ## use a counter
    count = 1

    ## use glob function to read each image in the file
    for filename in glob.glob(filepath + "/*.png"):
        print(filename)
        print(count)
        image = cv2.imread(filename)
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(destpath + "/" + str(count) + ".png", image)
        count += 1

def readJson(jsonfile):
    with open(jsonfile,encoding='utf-8') as f:
        jsonData = json.load(f)
    return jsonData

def rotate_bound(image, angle):
    """
    image rotation
    :param image: image
    :param angle: angle
    :return: the rotated image
    """
    h, w,_ = image.shape
    (cX, cY) = (w // 2, h // 2)
    print(cX,cY)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    image_rotate = cv2.warpAffine(image, M, (nW, nH),borderValue=(255,255,255))
    return image_rotate,cX,cY,angle


def dumpRotateImage(img, degree):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2
    print(width // 2,height // 2)
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    return imgRotation,matRotation


def rotate_xy(x, y, angle, cx, cy):
    """
    Point (x, y) rotates around point (cx, cy)
    """
    # print(cx,cy)
    angle = angle * pi / 180
    x_new = (x - cx) * cos(angle) - (y - cy) * sin(angle) + cx
    y_new = (x - cx) * sin(angle) + (y - cy) * cos(angle) + cy
    return x_new, y_new


def image_to_base64(image_np):
    image = cv2.imencode('.jpg', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code

def rotatePoint(Srcimg_rotate,jsonTemp,M,imagePath):
    json_dict = {}
    for key, value in jsonTemp.items():
        if key=='imageHeight':
            json_dict[key]=Srcimg_rotate.shape[0]
            print('Image height = ',json_dict[key])
        elif key=='imageWidth':
            json_dict[key] = Srcimg_rotate.shape[1]
            print('Image Width = ',json_dict[key])
        elif key=='imageData':
            json_dict[key] = image_to_base64(Srcimg_rotate)
        elif key=='imagePath':
            json_dict[key] = imagePath
        else:
            json_dict[key] = value
    for item in json_dict['shapes']:
        for key, value in item.items():
            if key == 'points':
                for item2 in range(len(value)):
                    pt1=np.dot(M,np.array([[value[item2][0]],[value[item2][1]],[1]]))
                    value[item2][0], value[item2][1] = pt1[0][0], pt1[1][0]
    return json_dict


def writeToJson(filePath,data):
    fb = open(filePath,'w')
    fb.write(json.dumps(data,indent=2)) # ,encoding='utf-8'
    fb.close()

def RESIZE(original_folder_path, new_folder_path, new_height, new_width, counter=10000):

    if not os.path.exists(new_folder_path):
        os.mkdir(new_folder_path)

    for filename in glob.glob(original_folder_path+"/*.png"):

        name, extension = filename.split('.')

        json_file = name + ".json"

        if not os.path.isfile(json_file): continue

        print("Start Reading " + str(name) + ".png")

        image = cv2.imread(name + ".png")
        height, width, channel = image.shape

        height_ratio = new_height / height
        width_ratio = new_width / width

        image_resize = cv2.resize(image, (new_width, new_height))
        cv2.imwrite(new_folder_path+ "/" +str(counter)+".png", image_resize)

        new_json_file = open(new_folder_path+ "/" +str(counter) +".json", "w")
        with open(json_file, 'rb') as f:
            annotation = json.load(f)
            for shape in annotation["shapes"]:
                point = np.array(shape["points"])
                print(point)
                temp = np.zeros(point.shape)
                temp[:, 0] = point[:, 0] * width_ratio
                temp[:, 1] = point[:, 1] * height_ratio
                temp = temp.tolist()
                shape["points"] = temp
            annotation['imageHeight'] = new_height
            annotation['imageWidth'] = new_width
            annotation['imagePath'] = str(counter)+ ".png"
            annotation['imageData'] = str(utils.img_arr_to_b64(image_resize[..., (2, 1, 0)]), encoding='utf-8')
            json.dump(annotation, new_json_file, indent=4)
        counter+=1

def Data_Rename(original_folder_path, new_folder_path, counter=10001):

    if not os.path.exists(new_folder_path):
        os.mkdir(new_folder_path)

    for filename in glob.glob(original_folder_path + "/*.png"):

        name, extension = filename.split('.')

        json_file = name + ".json"

        if not os.path.isfile(json_file): continue

        print("Start Reading " + str(name) + ".png")

        image = cv2.imread(name + ".png")
        height, width, channel = image.shape

        cv2.imwrite(new_folder_path+ "/" +str(counter)+".png", image)

        new_json_file = open(new_folder_path+ "/" +str(counter) +".json", "w")
        
        with open(json_file, 'rb') as f:
            annotation = json.load(f)
            annotation['imageData'] = str(utils.img_arr_to_b64(image[..., (2, 1, 0)]), encoding='utf-8')
            annotation['imagePath'] = str(counter)+ ".png"
            annotation['imageHeight'] = height
            annotation['imageWidth'] = width
            # annotation["version"] = "4.5.10"
            json.dump(annotation, new_json_file, indent=4)
        counter+=1

def Bounding_Box_Statistics(path, filename):
    with open(filename, 'w', newline='') as format:
        head = ['Filename', 'Defect_Type', 'Image_Width', 'Image_Height',
                'x_min', 'y_min', 'x_max', 'y_max', 'y_slope', 'x_slope', 'Area']
        Writer = csv.DictWriter(format, fieldnames=head, delimiter=',')

        Writer.writeheader()

        L = dict()

        for json_file in glob.glob(path+"/*.json"):
            with open(json_file, 'rb') as f:
                annotation = json.load(f)
                for shape in annotation["shapes"]:
                    point = np.array(shape["points"])
                    print(point)
                    L['y_min'] = np.min(point[:, 0])
                    L['y_max'] = np.max(point[:, 0])
                    L['x_min'] = np.min(point[:, 1])
                    L['x_max'] = np.max(point[:, 1])
                    L['Filename'] = annotation['imagePath']
                    L['Image_Width'] = annotation['imageWidth']
                    L['Image_Height'] = annotation['imageHeight']
                    L['Defect_Type'] = shape['label']
                    y_slope = (np.max(point[:, 0]) - np.min(point[:, 0])) / (np.max(point[:, 1]) - np.min(point[:, 1]))
                    L['y_slope'] = y_slope
                    x_slope = 1 / ((np.max(point[:, 0]) - np.min(point[:, 0])) / (
                                np.max(point[:, 1]) - np.min(point[:, 1])))
                    L['x_slope'] = x_slope
                    L['Area'] = np.sqrt(
                        (np.max(point[:, 0]) - np.min(point[:, 0])) * (np.max(point[:, 1]) - np.min(point[:, 1])))
                    Writer.writerow(L)
                    print(L)

def fold_move(json_file):
        with open(json_file, 'rb') as f:
                fold = json.load(f)['folds']
                f1 = fold['fold0']
                f2 = fold['fold1']
                f3 = fold['fold2']
                f4 = fold['fold3']

                for file in f1:
                        shutil.move("./Data/All/" + str(file) + ".png", "./Data/F1/" + str(file) + ".png")
                        shutil.move("./Data/All/" + str(file) + ".json", "./Data/F1/" + str(file) + ".json")

                for file in f2:
                        shutil.move("./Data/All/" + str(file) + ".png", "./Data/F2/" + str(file) + ".png")
                        shutil.move("./Data/All/" + str(file) + ".json", "./Data/F2/" + str(file) + ".json")

                for file in f3:
                        shutil.move("./Data/All/" + str(file) + ".png", "./Data/F3/" + str(file) + ".png")
                        shutil.move("./Data/All/" + str(file) + ".json", "./Data/F3/" + str(file) + ".json")

                for file in f4:
                        shutil.move("./Data/All/" + str(file) + ".png", "./Data/F4/" + str(file) + ".png")
                        shutil.move("./Data/All/" + str(file) + ".json", "./Data/F4/" + str(file) + ".json")

if __name__ == '__main__':
        fold_move("data_4folds_4fold_index.json")