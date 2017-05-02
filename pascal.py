import os
import xml.etree.ElementTree as ET
from utils import convert_bbox, print_class_counts, BoundingBox
import IO


class Pascal(object):
    """

    """
    def __init__(self, darknet):
        self.darknet = darknet
        self.source_dir = IO.data_source_dir
        self.name = 'pascal'
        # self.sets = [('2012', 'train'), ('2007', 'train')]
        self.sets = [('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val')]
        self.base_classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

        self.class_counts = None
        self.classes = self.load_classes()

    def load_classes(self):
        if self.darknet.classes is None:
            return {kls: i for i, kls in enumerate(self.base_classes)}

        classes = {}
        for label in self.base_classes:
            if label in self.darknet.classes:
                classes[label] = self.darknet.classes.index(label)
            elif label in self.darknet.aliases.keys():
                classes[label] = self.darknet.classes.index(self.darknet.aliases[label])

        self.class_counts = {label: 0 for label in classes.keys()}
        return classes

    def convert_annotation(self, year, image_id, classes_map):
        in_filename = '%s/%s/VOCdevkit/VOC%s/Annotations/%s.xml' % (self.source_dir, self.name, year, image_id)
        out_filename = '%s/%s/VOCdevkit/VOC%s/labels/%s.txt' % (self.darknet.labels_dir, self.name, year, image_id)

        in_file = open(in_filename)
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        class_bboxes = []
        for obj in root.iter('object'):
            difficult = obj.find('difficult')
            difficult = difficult.text if difficult is not None else 0
            cls = obj.find('name').text
            if cls not in classes_map or int(difficult) == 1:
                continue
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text),
                 float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            class_bboxes.append((cls, b))

        if len(class_bboxes) == 0:
            return False

        if not os.path.exists(out_filename.rpartition(os.sep)[0]):
            os.makedirs(out_filename.rpartition(os.sep)[0])

        out_file = open(out_filename, 'w')
        for cls, b in class_bboxes:
            cls_id = classes_map[cls]
            bb = convert_bbox((w, h), b)
            bbx = BoundingBox((w, h), b, 'voc').convert_to('darknet')
            print bb
            print bbx
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            self.class_counts[cls] += 1

        in_file.close()
        out_file.close()

        return True

    def create_darknet_dataset(self):

        for year, image_set in self.sets:
            self.source_imgdir = '%s/%s/VOCdevkit/VOC%s/JPEGImages' % (self.source_dir, self.name, year)
            image_ids = open('%s/%s/VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (self.source_dir, self.name, year, image_set)).read().strip().split()
            list_file = open('%s/%s_%s%s.list' % (self.darknet.temp_train_dir, self.name, year, image_set), 'w')
            for image_id in image_ids:
                src_img_filename = '%s/%s.jpg' % (self.source_imgdir, image_id)
                if not os.path.exists(src_img_filename):
                    continue  # no image file

                if self.convert_annotation(year, image_id, self.classes):
                    lnk_image_filename = src_img_filename.replace(self.source_dir, self.darknet.images_dir)
                    if not os.path.exists(lnk_image_filename.rpartition(os.sep)[0]):
                        os.makedirs(lnk_image_filename.rpartition(os.sep)[0])
                    os.symlink(src_img_filename, lnk_image_filename)
                    list_file.write(lnk_image_filename + '\n')

            list_file.close()

        print_class_counts(self.name, self.class_counts)


