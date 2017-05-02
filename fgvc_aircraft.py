import os
from utils import convert_bbox, print_class_counts, BoundingBox
import IO

LEVEL_HIERARCHY = {'manufacturer': 'manufacturers', 'family': 'families', 'variant': 'variants'}

# create a file with the image_ids of all the datasets:
# cat images_train.txt images_val.txt images_test.txt > images_trainvaltest.txt

# create a file in data/ with 3 columns: image_id, image_pixel_width, image_pixel_height and call it images_size.txt
# for F in `cat images_trainvaltest.txt`; do S=`convert images/$F -ping -format ' %[fx:w] %[fx:h]' info:`; echo $F $S; done > images_size.txt


class FGVC(object):
    """

    """
    def __init__(self, darknet, level):
        self.darknet = darknet
        self.name = 'fgvc-aircraft-2013b'
        self.source_dir = os.path.join(IO.data_source_dir, self.name, 'data')
        self.source_imgdir = os.path.join(self.source_dir, 'images')
        self.bbox_file = self.source_imgdir + '_box.txt'
        self.size_file = self.source_imgdir + '_size.txt'
        self.bboxes = {}
        self.sizes = {}

        # self.sets = [('2012', 'train'), ('2007', 'train')]
        self.sets = ['train', 'val']
        self.level = level
        self.classes_file = os.path.join(self.source_dir, LEVEL_HIERARCHY[level] + '.txt')
        comp_classes = {}
        with open(self.classes_file) as ifs:
            self.base_classes = ifs.read().strip().split('\n')

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

    def convert_annotation(self, image_id, cls):
        if cls not in self.classes:
            return False

        size = self.sizes[image_id]
        bbox = self.bboxes[image_id]
        bb = convert_bbox(size, bbox)
        bb_out = BoundingBox(size, bbox, 'fgvc').convert_to('darknet')
        print bb
        print bb_out

        out_filename = '%s/%s/%s.txt' % (self.darknet.labels_dir, self.name, image_id)
        if not os.path.exists(out_filename.rpartition(os.sep)[0]):
            os.makedirs(out_filename.rpartition(os.sep)[0])

        with open(out_filename, 'w') as ofs:
            ofs.write(str(self.classes[cls]) + " " + " ".join(map(str, bb_out)) + '\n')
            # ofs.write(str(cls_id) + " " + " ".join([str(a) for a in bb_out]) + '\n')

        self.class_counts[cls] += 1

        return True

    def create_darknet_dataset(self):

        with open(self.bbox_file) as ifs:
            lines = ifs.read().strip().split('\n')
            for line in lines:
                image_id, bbox = line.split(' ', 1)
                self.bboxes[image_id] = map(int, bbox.split())

        with open(self.sizes_file) as ifs:
            lines = ifs.read().strip().split('\n')
            for line in lines:
                image_id, size = line.split(' ', 1)
                self.sizes[image_id] = map(int, size.split())

        for image_set in self.sets:
            set_fname = '%s_%s_%s.txt' % (self.source_imgdir, self.level, image_set)
            links_fname = '%s/%s_%s.list' % (self.darknet.temp_train_dir, self.name, image_set)
            lines = open(set_fname).read().strip().split('\n')
            list_file = open(links_fname, 'w')
            for line in lines:
                image_id, cls = line.split(' ', 1)
                src_img_filename = '%s/%s.jpg' % (self.source_imgdir, image_id)
                if not os.path.exists(src_img_filename):
                    continue  # no image file

                if self.convert_annotation(image_id, cls):
                    lnk_image_filename = src_img_filename.replace(self.source_dir, self.darknet.images_dir)
                    if not os.path.exists(lnk_image_filename.rpartition(os.sep)[0]):
                        os.makedirs(lnk_image_filename.rpartition(os.sep)[0])
                    os.symlink(src_img_filename, lnk_image_filename)
                    list_file.write(lnk_image_filename + '\n')

            list_file.close()

        print_class_counts(self.name, self.class_counts)


