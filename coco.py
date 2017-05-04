import os
from pycocotools.coco import COCO
from utils import convert_coco_bbox, print_class_counts, BoundingBox
import IO


class Coco(object):
    """

    """
    def __init__(self, darknet):
        self.darknet = darknet
        self.name = 'coco'
        self.source_dir = os.path.join(IO.data_source_dir, self.name)
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

    def convert_annotation(self, out_filename, image_id, image_meta):

        annIds = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(annIds)

        class_bboxes = []
        for ann in anns:
            if ann['category_id'] not in self.classes:
                continue
            class_bboxes.append((ann['category_id'], ann['bbox']))

        if len(class_bboxes) == 0:
            return False

        # label_file = image_meta['file_name'].replace('jpg', 'txt')
        # label_filename = os.path.join(self.darknet.labels_dir, self.name, dataset, label_file)
        if not os.path.exists(out_filename.rpartition(os.sep)[0]):
            os.makedirs(out_filename.rpartition(os.sep)[0])

        out_file = open(out_filename, 'w')

        for cls, b in class_bboxes:
            # bb = convert_coco_bbox((image_meta['width'], image_meta['height']), b)
            bb = BoundingBox((image_meta['width'], image_meta['height']), b, 'coco').convert_to('darknet')
            out_file.write(str(self.classes[cls]) + " " + " ".join(map(str, bb)) + '\n')
            self.class_counts[cls] += 1

        out_file.close()

        return True

    def create_darknet_dataset(self):

        for dataset in self.datasets:

            annfile = os.path.join(self.source_dir, 'annotations', 'instances_' + dataset + '.json')
            self.coco = COCO(annfile)
            self.classes = self.load_classes()
            img_id_set = set()
            for cat_ids in self.classes.iterkeys():
                # cat_ids = self.coco.getCatIds(catNms=[kls])
                img_ids = self.coco.getImgIds(catIds=cat_ids)
                img_id_set = img_id_set.union(set(img_ids))
            self.img_ids = list(img_id_set)
            print '# of images to process:', len(self.img_ids)

            source_imgdir = os.path.join(self.source_dir, dataset)
            target_imgdir = os.path.join(self.darknet.images_dir, self.name, dataset)
            target_lbldir = os.path.join(self.darknet.labels_dir, self.name, dataset)
            list_file = open('%s/%s_%s.list' % (self.darknet.temp_train_dir, self.name, dataset), 'w')
            for image_id in self.img_ids:

                img = self.coco.loadImgs(image_id)[0]
                src_img_filename = os.path.join(source_imgdir, img['file_name'])
                if not os.path.exists(src_img_filename):
                    continue  # no image file

                label_file = img['file_name'].replace('jpg', 'txt')
                tgt_lbl_filename = os.path.join(target_lbldir, label_file)
                if self.convert_annotation(tgt_lbl_filename, image_id, img):
                    lnk_image_filename = os.path.join(target_imgdir, img['file_name'])
                    if not os.path.exists(lnk_image_filename.rpartition(os.sep)[0]):
                        os.makedirs(lnk_image_filename.rpartition(os.sep)[0])
                    os.symlink(src_img_filename, lnk_image_filename)
                    list_file.write(lnk_image_filename + '\n')

            list_file.close()

        print_class_counts(self.name, self.class_counts, mapper=self.classes_map)


