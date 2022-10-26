'''
rolabelimg xml data to dota 8 points data
'''
import os
import xml.etree.ElementTree as ET
import math
import cv2
import numpy as np
import shutil

ann_dir = r"C:\Graduate\mmdetection\data\SeaShips\VOC2007\Annotations"
img_dir = r"C:\Graduate\mmdetection\data\SeaShips\VOC2007\JPEGImages"
train = r"C:\Graduate\mmdetection\data\SeaShips\VOC2007\ImageSets\Main\train.txt"
test = r"C:\Graduate\mmdetection\data\SeaShips\VOC2007\ImageSets\Main\test.txt"
val = r"C:\Graduate\mmdetection\data\SeaShips\VOC2007\ImageSets\Main\val.txt"
trainval = r"C:\Graduate\mmdetection\data\SeaShips\VOC2007\ImageSets\Main\trainval.txt"

dst_train = r"C:\Graduate\mmrotate\data\ShipData\train"
dst_test = r"C:\Graduate\mmrotate\data\ShipData\test"
dst_val = r"C:\Graduate\mmrotate\data\ShipData\val"
dst_trainval = r"C:\Graduate\mmrotate\data\ShipData\trainval"

def edit_xml(xml_file, image_file, dest_path, file_name):
    if ".xml" not in xml_file:
        return
    tree = ET.parse(xml_file)
    objs = tree.findall('object')

    txt = dest_path + "\\annfiles\\" + file_name.replace("\n",".txt")
    dst_image = dest_path + "\\images\\" + file_name.replace("\n",".jpg")
    cmd = 'copy "%s" "%s"' % (image_file, dst_image)
    shutil.copyfile(image_file, dst_image)
    png = image_file
    src = cv2.imread(dst_image, 1)

    with open(txt, 'w') as wf:
        # wf.write("imagesource:Google\n")
        # wf.write("gsd:0.115726939386\n")

        for ix, obj in enumerate(objs):

            x0text = ""
            y0text = ""
            x1text = ""
            y1text = ""
            x2text = ""
            y2text = ""
            x3text = ""
            y3text = ""
            difficulttext = ""
            className = ""

            obj_name = obj.find('name')
            className = obj_name.text.replace(" ", "")

            obj_difficult = obj.find('difficult')
            if obj_difficult is None:
                difficulttext = 0
            else:
                difficulttext = obj_difficult.text
            obj_bnd = obj.find('bndbox')
            obj_xmin = obj_bnd.find('xmin')
            obj_ymin = obj_bnd.find('ymin')
            obj_xmax = obj_bnd.find('xmax')
            obj_ymax = obj_bnd.find('ymax')
            xmin = float(obj_xmin.text)
            ymin = float(obj_ymin.text)
            xmax = float(obj_xmax.text)
            ymax = float(obj_ymax.text)

            x0text = str(xmin)
            y0text = str(ymin)
            x1text = str(xmax)
            y1text = str(ymin)
            x2text = str(xmin)
            y2text = str(ymax)
            x3text = str(xmax)
            y3text = str(ymax)

            points = np.array([[int(float(x0text)), int(float(y0text))], [int(float(x1text)), int(float(y1text))], [int(float(x2text)), int(float(y2text))],
                               [int(float(x3text)), int(float(y3text))]], np.int32)
            cv2.polylines(src, [points], True, (255, 0, 0))  # 画任意多边
            # print(x0text,y0text,x1text,y1text,x2text,y2text,x3text,y3text,className,difficulttext)
            wf.write(
                "{} {} {} {} {} {} {} {} {} {}\n".format(x0text, y0text, x1text, y1text, x2text, y2text, x3text, y3text,
                                                         className, difficulttext))

        # cv2.imshow("ddd",src)
        # cv2.waitKey()


# 转换成四点坐标
def rotatePoint(xc, yc, xp, yp, theta):
    xoff = xp - xc;
    yoff = yp - yc;
    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = - sinTheta * xoff + cosTheta * yoff
    return str(int(xc + pResx)), str(int(yc + pResy))


if __name__ == '__main__':
    image_list = os.listdir(img_dir)
    ann_list = os.listdir(ann_dir)
    i = 0
    # with open(train, 'r') as f:
    #     for line in f.readlines():
    #         xml_file = os.path.join(ann_dir, line.replace("\n",".xml"))
    #         image_file = os.path.join(img_dir, line.replace("\n",".jpg"))
    #         edit_xml(xml_file, image_file, dst_train, line)
    #         i += 1
    #         print(i)
    # with open(test, 'r') as f:
    #     for line in f.readlines():
    #         xml_file = os.path.join(ann_dir, line.replace("\n",".xml"))
    #         image_file = os.path.join(img_dir, line.replace("\n",".jpg"))
    #         edit_xml(xml_file, image_file, dst_test, line)
    #         i += 1
    #         print(i)
    # with open(val, 'r') as f:
    #     for line in f.readlines():
    #         xml_file = os.path.join(ann_dir, line.replace("\n",".xml"))
    #         image_file = os.path.join(img_dir, line.replace("\n",".jpg"))
    #         edit_xml(xml_file, image_file, dst_val, line)
    #         i += 1
    #         print(i)
    with open(trainval, 'r') as f:
        for line in f.readlines():
            xml_file = os.path.join(ann_dir, line.replace("\n",".xml"))
            image_file = os.path.join(img_dir, line.replace("\n",".jpg"))
            edit_xml(xml_file, image_file, dst_trainval, line)
            i += 1
            print(i)