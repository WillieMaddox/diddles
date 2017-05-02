import re
import os
import xml.etree.ElementTree as ET
import IO
from utils import convert_bbox, print_class_counts


class Imagenet(object):
    """
    Image label summary files:
    <path-to-Imagenet>/devkit/data/

    classification - NO bounding boxes
    localization - has bounding boxes
    map_clsloc.txt
    1000 classes

    detection - has bounding boxes
    map_det.txt
    200 classes

    video - has bounding boxes
    map_vid.txt
    30 classes

    The format for the labels files is [label, index, name]:
    n02672831 1 accordion
    n02691156 2 airplane
    n02219486 3 ant

    names -> label -> ImageSets -> [Data, Annotation]

    grep -r --include=\*.xml n02958343 > ~/PycharmProjects/darknet/cars.txt
    """

    def __init__(self, darknet, extra_aliases_file=None):
        self.darknet = darknet
        self.extra_aliases_file = extra_aliases_file
        self.datadir = os.path.join(IO.data_source_dir, 'imagenet')
        self.name = 'ILSVRC'
        self.source_dir = os.path.join(self.datadir, self.name)
        self.source_anodir = os.path.join(self.source_dir, 'Annotations')
        self.source_imgdir = os.path.join(self.source_dir, 'Data')
        self.source_setsdir = os.path.join(self.source_dir, 'ImageSets')
        self.source_devkitdir = os.path.join(self.source_dir, 'devkit', 'data')

        self.classes_files = {
            'CLS-LOC': os.path.join(self.source_devkitdir, 'map_clsloc.txt'),
            'DET': os.path.join(self.source_devkitdir, 'map_det.txt'),
            # 'VID': os.path.join(self.source_devkitdir, 'map_vid.txt')
        }
        self.subcatcounts = 0
        self.aliases = {}
        self.class_counts = {
            'CLS-LOC': {},
            'DET': {}
        }
        self.classes = self.load_classes()

        self.sets = {
            'DET': {'train': ('',)},
        }

        # self.sets = {
        #     'CLS-LOC': {'train': ('_cls',)},
        #     'DET': {'train': ('',)},
        # }

        # self.sets = {
        #     'CLS-LOC': {'train': ('_cls', '_loc'), 'val': ('',)},
        #     'DET': {'train': ('',), 'val': ('',)},
        #     # 'VID': {'train': ('train.txt',), 'val': ('',)}
        # }

        self.blacklist = {
            'CLS-LOC': {'val': 'ILSVRC2015_clsloc_validation_blacklist.txt'},
            'DET': {'val': 'ILSVRC2015_det_validation_blacklist.txt'}
        }

        # self.synset_file = os.path.join(self.darknet.config['source_path'], 'imagenet.labels.list')
        # self.shortnames_file = os.path.join(self.darknet.config['source_path'], 'imagenet.shortnames.list')
        # self.synset_names_dict = self.gen_synset_names_dict()

        # self.synset_words_file = os.path.join(self.datadir, self.name, 'synset_words.txt')
        # self.synset_is_a_file = os.path.join(self.datadir, self.name, 'wordnet.is_a.txt')
        # self.wordnet_list = self.gen_wordnet_list()

    def load_classes(self):
        if self.extra_aliases_file:
            self.aliases = self.get_extra_aliases()

        if self.darknet.classes is None:
            return {}

        classes = {}
        for comp, classes_file in self.classes_files.iteritems():
            comp_classes = {}
            with open(classes_file) as ifs:
                classes_temp = ifs.read().strip().split('\n')
            base_classes = [kls.split() for kls in classes_temp]
            base_classes = {k: v for k, _, v in base_classes}

            for key, value in base_classes.iteritems():
                # if value in self.darknet.classes:
                #     comp_classes[key] = self.darknet.classes.index(value)
                #     print ' darknet.classes', key, value
                if key in self.aliases.keys():
                    if self.aliases[key] == 'person':
                        continue
                    comp_classes[key] = self.darknet.classes.index(self.aliases[key])
                    print 'extra_aliases', key, self.aliases[key]
                # elif value in self.darknet.aliases.keys():
                #     comp_classes[key] = self.darknet.classes.index(self.darknet.aliases[value])
                #     print ' darknet.aliases', key, value, self.darknet.aliases[value]

            for label in comp_classes.iterkeys():
                self.class_counts[comp][label] = 0

            classes[comp] = comp_classes

        return classes

    def get_extra_aliases(self):
        extra_aliases_temp = IO.load_txt(self.extra_aliases_file)
        label_synset_list = [line.split('\t') for line in extra_aliases_temp]
        extra_aliases = {}
        for label, synset_str in label_synset_list:
            synset_list = synset_str.split(',')
            for synset in synset_list:
                if synset in extra_aliases:
                    raise
                extra_aliases[synset] = label
        return extra_aliases

    def convert_annotation(self, comp, image_set, image_id, classes_map):
        out_filename = os.path.join(self.darknet.labels_dir, self.name, comp, image_set, image_id+'.txt')
        if os.path.exists(out_filename):
            return True

        in_filename = os.path.join(self.source_anodir, comp, image_set, image_id+'.xml')
        if not os.path.exists(in_filename):
            return False

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
            if int(difficult) == 1:
                continue

            name = obj.find('name').text
            if name not in classes_map:
                continue
                # sub = obj.find('subcategory')
                # subcat = sub.text if sub is not None else ''
                # if subcat not in classes_map:
                #     continue
                # else:
                #     cls = subcat
                #     self.subcatcounts += 1
            else:
                cls = name

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
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            self.class_counts[comp][cls] += 1

        in_file.close()
        out_file.close()
        return True

    def create_darknet_dataset(self):
        darknet_data_dir = os.path.join(self.darknet.images_dir, self.name)
        for comp, image_sets in self.sets.iteritems():
            classes = self.classes[comp]
            for image_set, image_subsets in image_sets.iteritems():

                if comp in self.blacklist and image_set in self.blacklist[comp]:
                    blacklist_filename = self.blacklist[comp][image_set]
                    blacklist_ids = open(os.path.join(self.source_devkitdir, blacklist_filename)).read().strip().split()
                else:
                    blacklist_ids = None

                darknet_file = '_'.join([self.name, comp, image_set + '.list'])
                darknet_listfile = os.path.join(self.darknet.temp_train_dir, darknet_file)
                list_file = open(darknet_listfile, 'w')

                for image_subset in image_subsets:
                    set_filename = os.path.join(self.source_setsdir, comp, image_set+image_subset+'.txt')
                    with open(set_filename) as ifs:
                        image_ids = ifs.read().strip().split('\n')

                    for image_id in image_ids:
                        if isinstance(image_id, str):
                            image_id = image_id.split()

                        # if isinstance(image_id, list):
                        assert len(image_id) == 2
                        if image_id[1] == -1:
                            continue  # negative image
                        if blacklist_ids is not None and image_id[0] in blacklist_ids:
                            continue  # blacklisted

                        image_id = image_id[0]

                        if image_id.find('/') >= 0:
                            cls_id = image_id.partition('/')[0]
                            if cls_id[0] == 'n' and cls_id[1:].isdigit() and cls_id not in classes:
                                continue  # no object in xml

                        src_img_filename = os.path.join(self.source_imgdir, comp, image_set, image_id + '.JPEG')
                        if not os.path.exists(src_img_filename):
                            continue  # no image file

                        if self.convert_annotation(comp, image_set, image_id, classes):

                            lnk_image_filename = src_img_filename.replace(self.source_imgdir, darknet_data_dir)
                            if os.path.exists(lnk_image_filename):
                                continue  # link already created.
                            if not os.path.exists(lnk_image_filename.rpartition(os.sep)[0]):
                                os.makedirs(lnk_image_filename.rpartition(os.sep)[0])
                            os.symlink(src_img_filename, lnk_image_filename)
                            list_file.write(lnk_image_filename + '\n')

                list_file.close()

            print_class_counts(self.name, self.class_counts[comp], mapper=self.aliases)
            print 'subcat counts: ', self.subcatcounts

    def old_code(self):
        dirs = set()

        imagekey = 'n02958343'
        for dir, subdirs, files in os.walk(self.anodir):
            for f in files:
                with open(os.path.join(dir, f)) as ifs:
                    for line in ifs:
                        if re.search(imagekey, line):
                            dirs.add(dir)

        with open('airplane.txt') as ifs:
            for line in ifs:
                p = re.compile("(.*)\/00")
                dir = p.search(line).group(1)
                dirs.add(dir)

        for dir in sorted(dirs):
            imagedir = os.path.join(self.imgdir, dir)
            imagefile = os.path.join(imagedir, "000000.JPEG")
            print(dir, len([name for name in os.listdir(imagedir) if os.path.isfile(os.path.join(imagedir, name))]))
            # img = cv2.imread(imagefile, 1)
            # cv2.imshow(dir, img)
            # cv2.waitKey()
        print(len(dirs))

    def gen_wordnet_list(self):
        synset_is_a = IO.load_txt(self.synset_is_a_file, col_delim=' ')
        synset_names = IO.load_txt(self.synset_words_file, col_delim='\t')
        synset_names_dict = {synset: name for synset, name in synset_names}
        for synset, name in self.synset_names_dict.iteritems():
            print synset
            print name
            print synset_names_dict[synset]
        names_list = []
        for parent, child in synset_is_a:
            if child not in synset_names_dict:
                print child
                continue
            if parent not in synset_names_dict:
                print parent
                continue
            names_list.append((synset_names_dict[parent], synset_names_dict[child]))
        IO.dump_pkl(os.path.join(self.darknet.config['source_path'], 'parent_child_words.pkl'), sorted(names_list))
        return names_list

    def gen_synset_names_dict(self):
        synsets = IO.load_txt(self.synset_file)
        names = IO.load_txt(self.shortnames_file)
        return {synset: name for synset, name in zip(synsets, names)}

    @staticmethod
    def get_name_subcategory_list(filename):
        name_subcategory_list = []
        in_file = open(filename)
        try:
            tree = ET.parse(in_file)
            root = tree.getroot()
        except ET.ParseError:
            print in_file
            return []

        for obj in root.iter('object'):
            subcategory = obj.find('subcategory')
            if subcategory is None:
                continue
            name = obj.find('name')
            if name is None:
                continue
            name_subcategory_list.append((name.text, subcategory.text))

        in_file.close()
        return name_subcategory_list

    def create_name_subcategory_file(self):
        name_subcategory_sets = set()
        root_dir = os.path.join(self.source_anodir, 'DET', 'train', 'ILSVRC2013_train')
        for dirpath, _, xml_filenames in os.walk(root_dir):
            for xml_filename in xml_filenames:
                file = os.path.join(dirpath, xml_filename)
                name_subcategory_list = self.get_name_subcategory_list(file)
                for name_subcategory in name_subcategory_list:
                    name_subcategory_sets.add(name_subcategory)

        synset_subcategory_dict = {}
        for name, subcategory in name_subcategory_sets:
            if name not in synset_subcategory_dict:
                synset_subcategory_dict[name] = []
            if subcategory not in synset_subcategory_dict[name]:
                synset_subcategory_dict[name].append(subcategory)

        out_filename = os.path.join(self.darknet.config['source_path'], 'imagenet_synset_subcategories.pkl')
        IO.dump_pkl(out_filename, synset_subcategory_dict)

        names_subcategory_dict = {}
        for synset_name, synset_subcategories in synset_subcategory_dict.iteritems():
            subcategory_list = []
            for synset_subcategory in synset_subcategories:
                subcategory_list.append(self.synset_names_dict[synset_subcategory])
            names_subcategory_dict[self.synset_names_dict[synset_name]] = subcategory_list

        out_filename = os.path.join(self.darknet.config['source_path'], 'imagenet_name_subcategories.pkl')
        IO.dump_pkl(out_filename, names_subcategory_dict)


