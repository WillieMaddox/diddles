import os
import time
import shutil
import networkx as nx
from subprocess import Popen, PIPE
from imagenet import Imagenet
from pascal import Pascal
from coco import Coco
from pixabay import Pixabay
from fgvc_aircraft import FGVC


class Darknet(object):
    def __init__(self, config_filename, clean=True):
        self.clean = clean
        self.config = {}
        self.model_subdirs = ['backup', 'JPEGImages', 'results', 'labels', 'temp_train', 'temp_valid']
        self.target_path = None
        self.backup_dir = None
        self.valid_res_dir = None
        self.images_dir = None
        self.labels_dir = None
        self.temp_train_dir = None
        self.temp_valid_dir = None
        self.labels_filename = None
        self.classes_filename = None
        self.darknet_train_filename = None
        self.darknet_valid_filename = None
        self.darknet_data_filename = None
        self.aliases_filename = None
        self.aliases = {}
        self.classes = None
        self.n_classes = None
        self.datasets = None
        self.synset_labels_dict = None
        self.synset_label_dict = None
        self.label_synsets_dict = None
        self.DAG = None

        self.load_config(config_filename)
        self.setup_model_directory()
        self.load_classes()
        self.get_aliases()
        self.build_model_directory()
        self.create_dataset_list()

    def load_config(self, config_filename):
        with open(config_filename) as ifs:
            config_lines = ifs.read().strip().split('\n')
        for line in config_lines:
            if line.startswith('#'):
                continue
            key, value = line.split('=')
            if value.find(',') >= 1:
                value = value.split(',')
            self.config[key] = value

        self.target_path = os.path.join(self.config['target_path'], self.config['model_name'])

    def setup_model_directory(self):
        for model_subdir in self.model_subdirs:
            subdir = os.path.join(self.target_path, model_subdir)
            if model_subdir == 'backup':
                self.backup_dir = subdir
            elif model_subdir == 'results':
                self.valid_res_dir = subdir
            elif model_subdir == 'JPEGImages':
                self.images_dir = subdir
            elif model_subdir == 'labels':
                self.labels_dir = subdir
            elif model_subdir == 'temp_train':
                self.temp_train_dir = subdir
            elif model_subdir == 'temp_valid':
                self.temp_valid_dir = subdir

        self.classes_filename = os.path.join(self.target_path, 'diddles.names')
        self.labels_filename = os.path.join(self.target_path, 'diddles.labels')
        self.darknet_train_filename = os.path.join(self.target_path, 'diddles.train.list')
        self.darknet_valid_filename = os.path.join(self.target_path, 'diddles.valid.list')
        self.darknet_data_filename = os.path.join(self.target_path, 'diddles.data')

    def load_classes(self):
        classes_filename = os.path.join(self.config['source_path'], self.config['model_name'] + '.names')
        with open(classes_filename) as ifs:
            self.classes = ifs.read().strip().split('\n')
        self.n_classes = len(self.classes)

    def get_aliases(self):
        if 'class_aliases_filename' not in self.config:
            self.aliases = {}
            return

        self.aliases_filename = os.path.join(self.config['source_path'], self.config['class_aliases_filename'])
        with open(self.aliases_filename) as ifs:
            aliases_list = ifs.read().strip().split('\n')
        label_alias_list = [line.split('\t') for line in aliases_list]
        self.aliases = {}
        for label, alias_str in label_alias_list:
            if label not in self.classes:
                continue
            alias_list = alias_str.split(',')
            for alias in alias_list:
                if alias in self.aliases:
                    raise
                self.aliases[alias] = label

        for cls in self.classes:
            if cls in self.aliases:
                continue
            self.aliases[cls] = cls

    def build_model_directory(self):
        for model_subdir in self.model_subdirs:
            subdir = os.path.join(self.target_path, model_subdir)
            if os.path.exists(subdir) and self.clean:
                shutil.rmtree(subdir)
            if not os.path.exists(subdir):
                os.makedirs(subdir)

        if self.clean:
            classes_filename = os.path.join(self.config['source_path'], self.config['model_name'] + '.names')
            shutil.copyfile(classes_filename, self.classes_filename)

            with open(self.labels_filename, 'w') as ofs:
                for i in range(self.n_classes):
                    ofs.write('%s\n' % str(i))

            with open(self.darknet_data_filename, 'w') as ofs:
                ofs.write('classes = %s\n' % self.n_classes)
                ofs.write('train = %s\n' % self.darknet_train_filename)
                ofs.write('valid = %s\n' % self.darknet_valid_filename)
                ofs.write('backup = %s\n' % self.backup_dir)
                ofs.write('results = %s\n' % self.valid_res_dir)
                ofs.write('labels = %s\n' % self.labels_filename)
                ofs.write('names = %s\n' % self.classes_filename)
                ofs.write('top = 5\n')
                ofs.write('eval = coco\n')

    def create_dataset_list(self):
        self.datasets = []
        if isinstance(self.config['datasets'], str):
            self.config['datasets'] = [self.config['datasets']]

        for dataset in self.config['datasets']:
            if dataset == '':
                continue
            elif dataset.lower() in ('imagenet', 'ilsvrc'):
                self.datasets.append('Imagenet')
            elif dataset.lower() in ('pascal', 'voc'):
                self.datasets.append('Pascal')
            elif dataset.lower() in ('coco', 'mscoco'):
                self.datasets.append('Coco')
            elif dataset.lower() in ('fgvc', 'fgvc_aircraft'):
                self.datasets.append('FGVC')
            elif dataset.lower() in ('pixabay',):
                self.datasets.append('Pixabay')
            else:
                raise

    def merge_datasets(self):
        temp_train_files = [os.path.join(self.temp_train_dir, f) for f in os.listdir(self.temp_train_dir)]
        if len(temp_train_files) == 0:
            return
        command = ['cat'] + temp_train_files
        p = Popen(command, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()
        if len(stderr) != 0:
            print stderr
            return
        with open(self.darknet_train_filename, 'w') as ofs:
            ofs.write(stdout)

        temp_valid_files = [os.path.join(self.temp_valid_dir, f) for f in os.listdir(self.temp_valid_dir)]
        if len(temp_valid_files) == 0:
            return
        command = ['cat'] + temp_valid_files
        p = Popen(command, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()
        if len(stderr) != 0:
            print stderr
            return
        with open(self.darknet_valid_filename, 'w') as ofs:
            ofs.write(stdout)

    def create_synset_labels_dict(self):
        if isinstance(self.synset_labels_dict, dict):
            return self.synset_labels_dict

        synset_words_file = "/media/SSD5/DATA/ILSVRC/synset_words.txt"
        with open(synset_words_file) as ifs:
            synsets = ifs.read().strip().split('\n')

        synsets = [line.split('\t') for line in synsets]
        slabels, swords = zip(*synsets)
        sword_lists = [w.split(',') for w in swords]

        swords1 = [[s.strip() for s in ss] for ss in sword_lists]

        t0 = time.time()
        self.synset_labels_dict = {s: l for s, l in zip(slabels, swords1)}
        self.synset_label_dict = {s: l[0] for s, l in zip(slabels, swords1)}
        print time.time() - t0

        t0 = time.time()
        self.label_synsets_dict = {}
        wordnet_inouns_file = "/home/maddoxw/nltk_data/corpora/wordnet/index.noun"
        with open(wordnet_inouns_file) as ifs:
            for line in ifs.readlines():
                if line.startswith('  '):
                    continue
                data = line.strip().split()
                ss = []
                for s in data[::-1]:
                    if len(s) < 8:
                        break
                    else:
                        ss.append('n'+s)
                assert len(ss) > 0
                self.label_synsets_dict[data[0]] = ss
        print time.time() - t0

    def create_word_graph(self):
        wordnet_DAG_file = "/media/SSD5/DATA/ILSVRC/wordnet.is_a.txt"
        self.DAG = nx.read_edgelist(wordnet_DAG_file, create_using=nx.DiGraph())

    def get_DAG_ordered_ancestors(self, child, ancestors):
        parents = self.DAG.predecessors(child)
        if len(parents) == 0:
            return []
        # elif len(parents) == 1:
        #     ancestors = self.get_DAG_ordered_ancestors(parents[0], ancestors)
        #     ancestors.append(str(parents[0]))
        # else:
        #     return ['!@!@!@!@!@!']
        ancestors = self.get_DAG_ordered_ancestors(parents[0], ancestors)
        ancestors.append(str(parents[0]))
        return ancestors

    def create_synset_aliases(self):
        t0 = time.time()
        self.create_word_graph()
        aliases_set = set([s for ss in self.aliases.itervalues() for s in ss])

        ofs = open('class_alias_ancestors.txt', 'w')

        # camap = {}
        for cls, aliases in self.aliases.iteritems():
            class_alias_ancestors = {}
            for synset, labels in self.synset_labels_dict.iteritems():
                if synset in class_alias_ancestors:
                    continue
                if not self.DAG.has_node(synset):
                    continue
                if len(set(aliases).intersection(labels)) == 0:
                    continue
                if len(aliases_set.intersection(labels)) == 0:
                    continue
                for alias in aliases:
                    if alias in labels:
                        ancestors = self.get_DAG_ordered_ancestors(synset, [])
                        print cls, alias, synset, len(ancestors)
                        # s = s.union(nx.ancestors(self.DAG, synset))
                        # print alias, synset, len(nx.ancestors(self.DAG, synset))
                        # camap[(cls, alias, synset)] = ancestors
                        class_alias_ancestors[synset] = ancestors[::-1]

            ofs.write(cls+'\n\n')
            for synset, ancestors in class_alias_ancestors.iteritems():
                ofs.write(', '.join([synset] + self.synset_labels_dict[synset]) + '\n')
                for ancestor in ancestors:
                    ofs.write(', '.join([ancestor] + self.synset_labels_dict[ancestor]) + '\n')
                ofs.write('\n')

        ofs.close()
        # for cls, aliases in self.aliases.iteritems():
        #     for alias in aliases:
        #     class_alias_sets[cls]
        # for label, synsets in self.label_synsets_dict.iteritems():
        #     if label in self.classes:
        #         for synset in synsets:
        #             if self.DG.has_node(synset):
        #                 print '{:15s}'.format(label), synset, self.synset_labels_dict[synset], len(nx.ancestors(self.DG, synset))
        # print time.time() - t0


if __name__ == '__main__':

    extra_labels_file = '/home/maddoxw/PycharmProjects/diddles/data/imagenet_name_synsets.txt'
    darknet = Darknet('/home/maddoxw/PycharmProjects/diddles/data/yolo_diddles.config')

    # extra_labels_file = '/home/maddoxw/PycharmProjects/diddles/data/imagenet_name_synsets.txt'
    # darknet = Darknet('/home/maddoxw/PycharmProjects/diddles/data/yolo_diddles.config')

    # darknet = Darknet('/home/maddoxw/PycharmProjects/diddles/data/coco', clean=False)

    # darknet.create_synset_labels_dict()
    # darknet.create_synset_aliases()

    # imagenet = Imagenet(darknet)
    # imagenet.create_name_subcategory_file()

    # pixabay = Pixabay(darknet)

    # md5old = '/media/RED6/DATA/pixabay/md5labels.txt'
    # md5new = '/media/maddoxw/Ativa/md5labels.txt'
    # pixabay.update_annotations(md5old, md5new)

    # pixabay.darknet_labels_to_xml()

    # target_classes = {'truck', 'jet', 'drone', 'helicopter', 'cannon'}
    # pixabay.preprocess(target_labels=target_classes)
    # pixabay.create_darknet_dataset()

    for ds in darknet.datasets:
        if ds == 'Imagenet':
            # imagenet = Imagenet(darknet, extra_labels_file)
            imagenet = Imagenet(darknet)
            imagenet.create_darknet_dataset()
        elif ds == 'Pascal':
            pascal = Pascal(darknet)
            pascal.create_darknet_dataset()
        elif ds == 'FGVC':
            fgvc = FGVC(darknet, 'family')
            fgvc.create_darknet_dataset()
        elif ds == 'Coco':
            coco = Coco(darknet)
            coco.create_darknet_dataset()
        elif ds == 'Pixabay':
            pixabay = Pixabay(darknet)
            pixabay.create_darknet_dataset()
        print ds, 'complete'

    darknet.merge_datasets()

    print 'done'
