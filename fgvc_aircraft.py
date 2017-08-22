import os
from utils import convert_bbox, print_class_counts, BoundingBox
import IO

LEVEL_HIERARCHY = {'manufacturer': 'manufacturers', 'family': 'families', 'variant': 'variants'}

# create a file with the image_ids of all the datasets:
# cat images_train.txt images_val.txt images_test.txt > images_trainvaltest.txt

# create a file in data/ with 3 columns: image_id, image_pixel_width, image_pixel_height and call it images_size.txt
# for F in `cat images_trainvaltest.txt`; do S=`convert images/$F.jpg -ping -format ' %[fx:w] %[fx:h] %[fx:z]' info:`; echo $F $S; done > images_size.txt


class FGVC(object):
    """

    """
    def __init__(self, darknet, level):
        self.darknet = darknet
        self.name = 'fgvc-aircraft-2013b'
        self.source_dir = os.path.join(IO.data_source_dir, 'FGVC', self.name, 'data')
        self.bboxes = {}
        self.sizes = {}

        self.sets = ['train', 'val', 'test']
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

    def convert_annotation(self, out_filename, image_id, cls):
        if cls not in self.classes:
            return False

        size = self.sizes[image_id]
        bbox = self.bboxes[image_id]
        # bb = convert_bbox(size, bbox)
        bb = BoundingBox(size, bbox, 'fgvc').convert_to('darknet')

        # out_filename = f'{self.darknet.labels_dir}/{self.name}/{image_id}.txt'
        if not os.path.exists(out_filename.rpartition(os.sep)[0]):
            os.makedirs(out_filename.rpartition(os.sep)[0])

        with open(out_filename, 'w') as ofs:
            ofs.write(str(self.classes[cls]) + " " + " ".join(map(str, bb)) + '\n')

        self.class_counts[cls] += 1

        return True

    def create_darknet_dataset(self):

        source_imgdir = os.path.join(self.source_dir, 'images')
        target_imgdir = os.path.join(self.darknet.images_dir, self.name)
        target_lbldir = os.path.join(self.darknet.labels_dir, self.name)
        with open(source_imgdir + '_box.txt') as ifs:
            lines = ifs.read().strip().split('\n')
            for line in lines:
                image_id, bbox = line.split(' ', 1)
                self.bboxes[image_id] = map(int, bbox.split())

        with open(source_imgdir + '_size.txt') as ifs:
            lines = ifs.read().strip().split('\n')
            for line in lines:
                image_id, size = line.split(' ', 1)
                self.sizes[image_id] = map(int, size.split())

        for image_set in self.sets:
            set_fname = f'{source_imgdir}_{self.level}_{image_set}.txt'
            links_fname = f'{self.darknet.temp_train_dir}/{self.name}_{image_set}.list'
            lines = open(set_fname).read().strip().split('\n')
            list_file = open(links_fname, 'w')
            for line in lines:
                image_id, cls = line.split(' ', 1)
                src_img_filename = os.path.join(source_imgdir, image_id + '.jpg')
                if not os.path.exists(src_img_filename):
                    continue  # no image file

                tgt_lbl_filename = os.path.join(target_lbldir, image_id + '.txt')
                if self.convert_annotation(tgt_lbl_filename, image_id, cls):
                    lnk_image_filename = os.path.join(target_imgdir, image_id + '.jpg')
                    if os.path.exists(lnk_image_filename):
                        list_file.write(lnk_image_filename + '\n')
                        continue  # skip existing
                    if not os.path.exists(os.path.dirname(lnk_image_filename)):
                        os.makedirs(os.path.dirname(lnk_image_filename))
                    os.symlink(src_img_filename, lnk_image_filename)
                    list_file.write(lnk_image_filename + '\n')

            list_file.close()

        print_class_counts(self.name, self.class_counts)


