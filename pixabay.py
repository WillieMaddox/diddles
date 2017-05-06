# from __future__ import unicode_literals
# import unicodedata
import os
import shutil
import json
import xml.etree.ElementTree as ET
from lxml import etree
from subprocess import Popen, PIPE
import numpy as np
from utils import convert_bbox, print_class_counts
import IO


class Pixabay(object):
    """

    """

    def __init__(self, darknet):
        self.darknet = darknet
        self.name = 'pixabay'
        self.source_dir = IO.data_source_dir
        self.datadir = os.path.join(self.source_dir, self.name)
        self.ano_dir = os.path.join(self.datadir, 'Annotations')
        self.img_dir = os.path.join(self.datadir, 'JPEGImages')
        self.temp_ano_dir = os.path.join(self.datadir, 'temp_Anno')
        self.temp_img_dir = os.path.join(self.datadir, 'temp_Imag')
        # self.valid_img_dir = '/home/maddoxw/git/darknet/data/diddles/models/office30/valid_img'
        self.valid_xml_dir = os.path.join(self.darknet.target_path, 'valid_xml')
        if not os.path.exists(self.valid_xml_dir):
            os.makedirs(self.valid_xml_dir)
        # self.valid_res_dir = '/home/maddoxw/git/darknet/data/diddles/models/office30/valid_res'
        self.min_required = 1
        # self.classes = ['person', 'boat', 'automobile', 'airplane', 'bike']
        self.extra_aliases_dual = {
            'person': ['person', 'woman', 'people', 'man', 'girl', 'boy', 'male', 'female', 'human'],
            'boat': ['boat', 'boats', 'ship', 'sailboat', 'sailboats'],
            'airplane': ['airplane', 'aeroplane', 'plane', 'jet', 'aircraft'],
            'bike': ['bike', 'bicycle', 'motorbike', 'motorcycle', 'bicycles']
        }
        self.aliases = {}
        self.class_counts = None
        self.classes = self.load_classes()

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        try:
            rough_string = ET.tostring(elem, 'utf8')
        except Exception:
            print elem
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True)

    def gen_xml(self, img_id, wid, hgt):
        """
            Return XML root
        """
        top = ET.Element('annotation')

        unprocessed = ET.SubElement(top, 'unprocessed')
        unprocessed.text = 'True'

        folder = ET.SubElement(top, 'folder')
        folder.text = 'images'

        filename = ET.SubElement(top, 'filename')
        filename.text = str(img_id) + '.jpg'

        source = ET.SubElement(top, 'source')
        database = ET.SubElement(source, 'database')
        database.text = self.name

        size_part = ET.SubElement(top, 'size')
        width = ET.SubElement(size_part, 'width')
        width.text = str(wid)
        height = ET.SubElement(size_part, 'height')
        height.text = str(hgt)
        depth = ET.SubElement(size_part, 'depth')
        depth.text = '3'

        segmented = ET.SubElement(top, 'segmented')
        segmented.text = '0'
        return top

    def append_objects(self, top, valid_labels, boxlist, scores=None):
        for i, label in enumerate(valid_labels):
            object_item = ET.SubElement(top, 'object')
            name = ET.SubElement(object_item, 'name')
            name.text = str(label)
            if scores is not None:
                score = ET.SubElement(object_item, 'score')
                score.text = str(scores[i])
            pose = ET.SubElement(object_item, 'pose')
            pose.text = "Unspecified"
            truncated = ET.SubElement(object_item, 'truncated')
            truncated.text = "0"
            difficult = ET.SubElement(object_item, 'difficult')
            difficult.text = "0"
            bndbox = ET.SubElement(object_item, 'bndbox')
            for key, value in boxlist[i].iteritems():
                box_element = ET.SubElement(bndbox, key)
                box_element.text = str(int(value))

    def save_xml(self, root, filename):
        prettify_result = self.prettify(root)
        with open(filename, 'w') as ofs:
            ofs.write(prettify_result)

    def gen_random_bboxes(self, n, meta):
        """
        w = (n+1) * aw + n * bw
        bw = 3 * aw
        h = (n+1) * ah + n * bh
        bh = 3 * ah
        :param n: number of bboxes to generate
        :type n: int
        :param meta: image metadata. Should have height and width keys.
        :type meta: dict
        :return: list of bboxes. [{xmin: xmin, xmax: xmax, ymin: ymin, ymax: ymax}, {xmin: xmin, ...}, ... {}}
        :rtype: list of dicts
        """
        def get_mins_and_maxs(d):
            a = int(d / (4 * grid_dim + 1))
            b = 3 * a
            mins = []
            maxs = []
            for i in range(grid_dim):
                mn = (1 + i) * a + i * b
                mx = mn + b
                mins.append(mn)
                maxs.append(mx)
            return mins, maxs

        grid_dim = int(np.ceil(np.sqrt(n)))
        xmins, xmaxs = get_mins_and_maxs(meta['width'])
        ymins, ymaxs = get_mins_and_maxs(meta['height'])

        bboxes = []
        for ymin, ymax in zip(ymins, ymaxs):
            for xmin, xmax in zip(xmins, xmaxs):
                bboxes.append({'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax})

        return bboxes

    def preprocess(self, hit_limit=2, target_labels=None):
        metadata = IO.read_pixabay_metadata_file()
        if target_labels is None:
            tallys = IO.read_pixabay_tally_file(hit_limit)
            target_labels = set(tallys.keys())

        for img_id, img_meta in metadata.iteritems():

            valid_labels = target_labels.intersection(img_meta['tags'])
            if len(valid_labels) == 0:
                continue

            img_filename = os.path.join(self.img_dir, str(img_id) + '.jpg')
            if not os.path.exists(img_filename):
                continue

            temp_ano_filename = os.path.join(self.temp_ano_dir, str(img_id) + '.xml')
            if os.path.exists(temp_ano_filename):
                continue

            boxlist = self.gen_random_bboxes(len(valid_labels), img_meta)
            root = self.gen_xml(img_id, img_meta['width'], img_meta['height'])
            self.append_objects(root, valid_labels, boxlist)
            self.save_xml(root, temp_ano_filename)

            temp_img_filename = os.path.join(self.temp_img_dir, str(img_id) + '.jpg')
            if not os.path.exists(temp_img_filename):
                os.symlink(img_filename, temp_img_filename)

    def over_classification_filter(self, in_metadata, max_classes=30):
        """
        Filter out images that have a large number of classification labels.
        Images with lots of labels its usually a good indication that many of the labels are wrong.

        :param in_metadata:
        :type in_metadata:
        :param max_classes:
        :type max_classes:
        :return:
        :rtype:
        """
        print 'size before:', len(in_metadata)
        out_metadata = {}
        for img_id, img_dict in in_metadata.iteritems():
            if img_dict['n_labels'] <= max_classes:
                out_metadata[img_id] = img_dict

        print 'size after:', len(out_metadata)
        return out_metadata

    def unbias_dataset(self, in_metadata, cls_counts, max_classes):
        """
        restrict the number of labeled images such that all classes have roughly the same number of images.
        For example, We don't want to bias the training with 40000 people when we only have 5000 airplanes.
        So, pick 5000 of the best people images.

        :param in_metadata:
        :type in_metadata:
        :param cls_counts:
        :type cls_counts:
        :param max_classes:
        :type max_classes:
        :return:
        :rtype:
        """

        for kls, cts in cls_counts.iteritems():
            if cts < max_classes:
                max_classes = cts

        out_metadata = {}
        targeted_counts = {kls: 0 for kls in self.classes}
        for img_id, img_dict in sorted(in_metadata.iteritems(), key=lambda x: x[1]['hit_score'], reverse=True):
            good_image = False
            for label in img_dict['labels']:
                if targeted_counts[label] < max_classes:
                    targeted_counts[label] += 1
                    good_image = True
            if good_image:
                out_metadata[img_id] = img_dict

        return out_metadata

    def get_bounding_boxes(self, in_metadata):
        """

        :param in_metadata:
        :type in_metadata:
        :return:
        :rtype:
        """

        base_command = ['identify', '-format', '%[fx:w],%[fx:h]']
        for img_id, img_dict in in_metadata.iteritems():
            if 'bbox' in img_dict:
                continue
            command = base_command + [self.datadir + 'images/' + str(img_id) + '.jpg']
            p = Popen(command, stdout=PIPE, stderr=PIPE)
            stdout, stderr = p.communicate()
            if len(stderr) != 0:
                continue
            w, h = map(int, stdout.split(','))
            img_dict['bbox'] = [int(w / 2), int(h / 2), w, h]

        return in_metadata

    def load_classes(self):
        self.aliases = self.get_aliases()
        classes = {}
        for label in self.darknet.classes:
            if label in self.aliases.keys():
                classes[label] = self.darknet.classes.index(self.aliases[label])

        self.class_counts = {label: 0 for label in classes.keys()}
        return classes

    def get_aliases(self):
        aliases = {}
        for label, alias_list in self.extra_aliases_dual.iteritems():
            for alias in alias_list:
                if alias in aliases:
                    continue
                aliases[alias] = label

        for alias, label in self.darknet.aliases.iteritems():
            if alias in aliases:
                continue
            aliases[alias] = label

        return aliases

    def convert_annotation(self, image_id, classes_map):
        out_filename = os.path.join(self.darknet.labels_dir, self.name, 'labels', image_id + '.txt')
        if os.path.exists(out_filename):
            return True

        in_filename = os.path.join(self.ano_dir, image_id + '.xml')
        if not os.path.exists(in_filename):
            return False

        in_file = open(in_filename)
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        d = int(size.find('depth').text)
        if d != 3:
            print image_id, ' is greyscale. Skipping.'
            return False

        w = float(size.find('width').text)
        h = float(size.find('height').text)
        if w < 1 or h < 1:
            print image_id, 'width and/or height == 0'
            return False

        class_bboxes = []
        for obj in root.iter('object'):
            difficult = obj.find('difficult')
            difficult = difficult.text if difficult is not None else 0
            if int(difficult) == 1:
                print image_id, 'difficult == 1'
                continue
            alias = obj.find('name').text
            if alias not in self.aliases:
                print image_id, 'skipping label', alias
                continue
            cls = self.aliases[alias]
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text),
                 float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            class_bboxes.append((cls, b))

        if len(class_bboxes) == 0:
            print image_id, 'no bounding boxes detected'
            return False

        if not os.path.exists(out_filename.rpartition(os.sep)[0]):
            os.makedirs(out_filename.rpartition(os.sep)[0])

        out_file = open(out_filename, 'w')
        for cls, b in class_bboxes:
            bb = convert_bbox((w, h), b)
            out_file.write(str(classes_map[cls]) + " " + " ".join(map(str, bb)) + '\n')
            self.class_counts[cls] += 1

        in_file.close()
        out_file.close()
        return True

    def create_darknet_dataset(self):

        target_imgdir = os.path.join(self.darknet.images_dir, self.name)
        list_file = open('%s/%s.list' % (self.darknet.temp_train_dir, self.name), 'w')

        for dirpath, _, xml_filenames in os.walk(self.ano_dir):
            for xml_fname in xml_filenames:
                image_id = os.path.splitext(xml_fname)[0]
                img_file = image_id + '.jpg'
                src_img_filename = os.path.join(self.img_dir, img_file)
                if not os.path.exists(src_img_filename):
                    print image_id, 'image is missing.'
                    continue  # no image file

                if self.convert_annotation(image_id, self.classes):
                    lnk_image_filename = os.path.join(target_imgdir, img_file)
                    if not os.path.exists(lnk_image_filename.rpartition(os.sep)[0]):
                        os.makedirs(lnk_image_filename.rpartition(os.sep)[0])
                    os.symlink(src_img_filename, lnk_image_filename)
                    list_file.write(lnk_image_filename + '\n')

        list_file.close()

        print_class_counts(self.name, self.class_counts, skip_zeros=True)

    # def create_validation_dataset(self, size=100, pattern='random', classes='any'):
    #     metadata = IO.read_pixabay_metadata_file()
    #     if size > 0:
    #         if pattern == 'random':
    #             pass

    def gen_valid_bboxes(self, records, names):
        """
        Run this after ./darknet detector valid ...
        :param records:
        :type records:
        :param names:
        :type names:
        :return:
        :rtype:
        """
        labels = []
        boxes = []
        scores = []
        for rec in records:
            scores.append(rec['score'])
            labels.append(names[rec['category_id'] - 1])
            bx, by, bw, bh = rec['bbox']
            boxes.append({'xmin': bx, 'ymin': by, 'xmax': bx+bw, 'ymax': by+bh})

        return labels, boxes, scores

    def darknet_labels_to_xml(self):

        lbl_meta = {}
        res_filename = os.path.join(self.darknet.valid_res_dir, 'coco_results.json')
        with open(res_filename) as ifs:
            recs = json.load(ifs)

        for rec in recs:
            if rec['score'] <= 0.2:
                continue
            image_id = rec.pop('image_id')
            if image_id not in lbl_meta:
                lbl_meta[image_id] = []
            lbl_meta[image_id].append(rec)

        img_meta = IO.read_pixabay_metadata_file()
        names = IO.load_txt(self.darknet.classes_filename)

        for img_id, rec in lbl_meta.iteritems():

            img_filename = os.path.join(self.darknet.data_dir, str(img_id) + '.jpg')
            if not os.path.exists(img_filename):
                continue

            xml_filename = os.path.join(self.valid_xml_dir, str(img_id) + '.xml')
            if os.path.exists(xml_filename):
                continue

            root = self.gen_xml(img_id, img_meta[img_id]['width'], img_meta[img_id]['height'])
            labels, bboxes, scores = self.gen_valid_bboxes(rec, names)
            self.append_objects(root, labels, bboxes, scores)
            self.save_xml(root, xml_filename)

    def update_annotations(self, md5file_old, md5file_new):
        """
        :param md5file_old: filename of md5sum generated file for unprocessed xmls
        :type md5file_old: string
        :param md5file_new: filename of md5sum generated file for processed xmls
        :type md5file_new: string
        :return:
        :rtype:
        REQUIREMENTS:
            1. The xml filenames MUST be formated with a fully qualified path.
            2. The xml filenames MUST contain NO spaces.
            3. All old xml filenames MUST have the SAME root path.
            4. All new xml filenames MUST have the SAME root path.
            5. old and new xml filenames MUST have different paths.
            6. both md5 files MUST have the same number of items.

        example:
        Run something like this from the command line:
        maddoxw@wmdevbox:/media/RED6/DATA/pixabay$ for f in `ls /media/RED6/DATA/pixabay/temp_Anno`; do echo `md5sum /media/RED6/DATA/pixabay/temp_Anno/$f`; done > /media/RED6/DATA/pixabay/md5labels.txt
        old file
        6e92202cc6fa5947dedad186b3855ecd /media/RED6/DATA/pixabay/temp_Anno/1000870.xml
        96ed594a308f2dc75bae5491a7b24ab8 /media/RED6/DATA/pixabay/temp_Anno/1000.xml

        new file
        6e92202cc6fa5947dedad186b3855ecd /media/maddoxw/Ativa/temp_Anno/1000870.xml
        07a2d891421bed4d12d23c00e7f886f7 /media/maddoxw/Ativa/temp_Anno/1000.xml

        1000870.xml is the same so don't copy to Annotations dir.
        1000.xml is different. Copy to Annotations dir.

        """

        md5old_list = IO.load_txt(md5file_old, col_delim=' ')
        md5new_list = IO.load_txt(md5file_new, col_delim=' ')
        assert len(md5old_list) == len(md5new_list)

        md5old_dict = {os.path.basename(filename): (md5, filename) for md5, filename in md5old_list}
        md5new_dict = {os.path.basename(filename): (md5, filename) for md5, filename in md5new_list}

        for fname, (md5_new, filename_new) in md5new_dict.iteritems():
            md5_old, filename_old = md5old_dict[fname]
            if md5_old == md5_new:
                continue
            filename_ano = os.path.join(self.ano_dir, fname)
            shutil.copyfile(filename_new, filename_ano)


