import os
from pycocotools.coco import COCO
from utils import convert_coco_bbox, print_class_counts
import IO


class Coco(object):
    """

    """
    def __init__(self, darknet):
        self.darknet = darknet
        self.source_dir = IO.data_source_dir
        self.name = 'coco'
        self.datasets = ('train2014', 'val2014')
        self.coco = None
        self.classes = None
        self.img_ids = None
        self.class_counts = {}
        self.classes_map = {}

    def load_classes(self):

        if self.darknet.classes is None:
            return {}

        cats = self.coco.loadCats(self.coco.getCatIds())
        base_classes = [cat['name'] for cat in cats]

        classes = {}
        self.classes_map = {}
        for i, label in enumerate(base_classes):
            if label in self.darknet.classes:
                classes[i+1] = self.darknet.classes.index(label)
                self.classes_map[i+1] = label
            elif label in self.darknet.aliases.keys():
                classes[i+1] = self.darknet.classes.index(self.darknet.aliases[label])
                self.classes_map[i+1] = self.darknet.aliases[label]
            else:
                continue

        self.class_counts = {label: 0 for label in classes.keys()}
        return classes

    def convert_annotation(self, datatype, image_id, image_meta):

        label_file = image_meta['file_name'].replace('jpg', 'txt')
        label_filename = os.path.join(self.darknet.labels_dir, self.name, datatype, label_file)

        annIds = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(annIds)

        class_bboxes = []
        for ann in anns:
            if ann['category_id'] not in self.classes:
                continue
            class_bboxes.append((ann['category_id'], ann['bbox']))

        if len(class_bboxes) == 0:
            return False

        if not os.path.exists(label_filename.rpartition(os.sep)[0]):
            os.makedirs(label_filename.rpartition(os.sep)[0])

        out_file = open(label_filename, 'w')

        for cls, b in class_bboxes:
            cls_id = self.classes[cls]
            bb = convert_coco_bbox((image_meta['width'], image_meta['height']), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            self.class_counts[cls] += 1

        out_file.close()

        return True

    def create_darknet_dataset(self):

        for dataset in self.datasets:

            annfile = '%s/%s/annotations/instances_%s.json' % (self.source_dir, self.name, dataset)
            self.coco = COCO(annfile)
            self.classes = self.load_classes()
            img_id_set = set()
            for cat_ids in self.classes.iterkeys():
                # cat_ids = self.coco.getCatIds(catNms=[kls])
                img_ids = self.coco.getImgIds(catIds=cat_ids)
                img_id_set = img_id_set.union(set(img_ids))
            self.img_ids = list(img_id_set)

            source_imgdir = os.path.join(self.source_dir, self.name, dataset)
            list_file = open('%s/%s_%s.list' % (self.darknet.temp_train_dir, self.name, dataset), 'w')
            print 'n_images to process:', len(self.img_ids)
            for image_id in self.img_ids:
                img = self.coco.loadImgs(image_id)[0]

                src_img_filename = os.path.join(source_imgdir, img['file_name'])
                if not os.path.exists(src_img_filename):
                    continue  # no image file

                if self.convert_annotation(dataset, image_id, img):
                    lnk_image_filename = src_img_filename.replace(self.source_dir, self.darknet.images_dir)
                    if not os.path.exists(lnk_image_filename.rpartition(os.sep)[0]):
                        os.makedirs(lnk_image_filename.rpartition(os.sep)[0])
                    os.symlink(src_img_filename, lnk_image_filename)
                    list_file.write(lnk_image_filename + '\n')

            list_file.close()

        print_class_counts(self.name, self.class_counts, mapper=self.classes_map)


