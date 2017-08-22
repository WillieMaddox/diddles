# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET
from pycocotools.coco import COCO
from utils import convert_bbox, convert_coco_bbox, BoundingBox
import IO

# Original code @ferada http://codereview.stackexchange.com/questions/128315/k-means-clustering-algorithm-implementation


def area(x):
    if len(x.shape) == 1:
        return x[0] * x[1]
    else:
        return x[:, 0] * x[:, 1]


def kmeans_iou(k, centroids, points, iter_count=0, iteration_cutoff=25, feature_size=13):

    best_clusters = []
    best_avg_iou = 0
    best_avg_iou_iteration = 0

    npoi = points.shape[0]
    area_p = area(points)  # (npoi, 2) -> (npoi,)

    while True:
        cen2 = centroids.repeat(npoi, axis=0).reshape(k, npoi, 2)
        cdiff = points - cen2
        cidx = np.where(cdiff < 0)
        cen2[cidx] = points[cidx[1], cidx[2]]

        wh = cen2.prod(axis=2).T  # (k, npoi, 2) -> (npoi, k)
        dist = 1. - (wh / (area_p[:, np.newaxis] + area(centroids) - wh))  # -> (npoi, k)
        belongs_to_cluster = np.argmin(dist, axis=1)  # (npoi, k) -> (npoi,)
        clusters_niou = np.min(dist, axis=1)  # (npoi, k) -> (npoi,)
        clusters = [points[belongs_to_cluster == i] for i in range(k)]
        avg_iou = np.mean(1. - clusters_niou)
        if avg_iou > best_avg_iou:
            best_avg_iou = avg_iou
            best_clusters = clusters
            best_avg_iou_iteration = iter_count

        print(f"\nIteration {iter_count}")
        print(f"Average iou to closest centroid = {avg_iou}")
        print(f"Sum of all distances (cost) = {np.sum(clusters_niou)}")

        new_centroids = np.array([np.mean(c, axis=0) for c in clusters])
        isect = np.prod(np.min(np.asarray([centroids, new_centroids]), axis=0), axis=1)
        aa1 = np.prod(centroids, axis=1)
        aa2 = np.prod(new_centroids, axis=1)
        shifts = 1 - isect / (aa1 + aa2 - isect)

        # for i, s in enumerate(shifts):
        #     print("{}: Cluster size: {}, Centroid distance shift: {}".format(i, len(clusters[i]), s))

        if sum(shifts) == 0 or iter_count >= best_avg_iou_iteration + iteration_cutoff:
            break

        centroids = new_centroids
        iter_count += 1

    # Get anchor boxes from best clusters
    anchors = np.asarray([np.mean(cluster, axis=0) for cluster in best_clusters])
    anchors = anchors[anchors[:, 0].argsort()]
    print(f"k-means clustering pascal anchor points (original coordinates) \
    \nFound at iteration {best_avg_iou_iteration} with best average IoU: {best_avg_iou} \
    \n{anchors*feature_size}")

    return anchors


def plot_anchors(pascal_anchors, coco_anchors):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.set_ylim([0, 500])
    ax1.set_xlim([0, 900])

    for i in range(len(pascal_anchors)):

        if area(pascal_anchors[i]) > area(coco_anchors[i]):
            bbox1 = pascal_anchors[i]
            color1 = "green"
            bbox2 = coco_anchors[i]
            color2 = "blue"
        else:
            bbox1 = coco_anchors[i]
            color1 = "blue"
            bbox2 = pascal_anchors[i]
            color2 = "green"

        lrx = bbox1[0]-(bbox1[2]/2.0)
        lry = bbox1[1]-(bbox1[3]/2.0)
        ax1.add_patch(patches.Rectangle((lrx, lry), bbox1[2], bbox1[3], facecolor=color1))

        lrx = bbox2[0]-(bbox2[2]/2.0)
        lry = bbox2[1]-(bbox2[3]/2.0)
        ax1.add_patch(patches.Rectangle((lrx, lry), bbox2[2], bbox2[3], facecolor=color2))

    plt.show()


def load_fgvc_dataset():
    name = 'fgvc-aircraft-2013b'
    data = []
    bboxes = {}
    sizes = {}

    source_dir = os.path.join(IO.data_source_dir, 'FGVC', name, 'data')
    source_imgdir = os.path.join(source_dir, 'images')

    with open(source_imgdir + '_box.txt') as ifs:
        lines = ifs.read().strip().split('\n')
        for line in lines:
            image_id, bbox = line.split(' ', 1)
            bboxes[image_id] = list(map(int, bbox.split()))

    with open(source_imgdir + '_size.txt') as ifs:
        lines = ifs.read().strip().split('\n')
        for line in lines:
            image_id, size = line.split(' ', 1)
            sizes[image_id] = list(map(int, size.split()))

    for key in bboxes.keys():
        size = sizes[key]
        bbox = bboxes[key]
        bb = BoundingBox(size, bbox, 'fgvc').convert_to('darknet')
        data.append(bb[2:])

    return np.array(data)


def load_pascal_dataset():
    name = 'pascal'
    data = []

    for year, image_set in datasets:
        img_ids_filename = f'{source_dir}/{name}/VOCdevkit/VOC{year}/ImageSets/Main/{image_set}.txt'
        ifs_img_ids = open(img_ids_filename)
        img_ids = ifs_img_ids.read().strip().split()

        for image_id in img_ids:
            anno_filename = f'{source_dir}/{name}/VOCdevkit/VOC{year}/Annotations/{image_id}.xml'
            ifs_anno = open(anno_filename)
            tree = ET.parse(ifs_anno)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            for obj in root.iter('object'):
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text),
                     float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text),
                     float(xmlbox.find('ymax').text))
                bb = convert_bbox((w, h), b)
                data.append(bb[2:])

            ifs_anno.close()
        ifs_img_ids.close()

    return np.array(data)


def load_coco_dataset():
    name = 'coco'
    data = []

    for dataset in datasets:
        annfile = f'{source_dir}/{name}/annotations/instances_{dataset}.json'
        coco = COCO(annfile)
        cats = coco.loadCats(coco.getCatIds())
        base_classes = {cat['id']: cat['name'] for cat in cats}
        img_id_set = set()

        for cat_ids in iter(base_classes.keys()):
            img_ids = coco.getImgIds(catIds=cat_ids)
            img_id_set = img_id_set.union(set(img_ids))
        image_ids = list(img_id_set)

        for image_id in image_ids:
            annIds = coco.getAnnIds(imgIds=image_id)
            anns = coco.loadAnns(annIds)
            img = coco.loadImgs(image_id)[0]
            w = img['width']
            h = img['height']

            for ann in anns:
                b = ann['bbox']
                bb = convert_coco_bbox((w, h), b)
                data.append(bb[2:])

    return np.array(data)

if __name__ == "__main__":

    # examples
    # k, pascal, coco
    # 1, 0.30933335617, 0.252004954777
    # 2, 0.45787906725, 0.365835079771
    # 3, 0.53198291772, 0.453180358467
    # 4, 0.57562962803, 0.500282182136
    # 5, 0.58694643198, 0.522010174068
    # 6, 0.61789602056, 0.549904351137
    # 7, 0.63443906479, 0.569485509501
    # 8, 0.65114747974, 0.585718648162
    # 9, 0.66393113546, 0.601564171461

    # k-means picking the first k points as centroids
    img_size = 416
    k = 5

    # random_data = np.random.random((1000, 2))
    # centroids = np.random.random((k, 2))
    # random_anchors = kmeans_iou(k, centroids, random_data)

    source_dir = IO.data_source_dir

    datasets = ('train', 'val', 'test')
    fgvc_data = load_fgvc_dataset()
    centroids = fgvc_data[np.random.choice(np.arange(len(fgvc_data)), k, replace=False)]
    fgvc_anchors = kmeans_iou(k, centroids, fgvc_data, feature_size=img_size / 32)

    # datasets = (('2007', 'train'), ('2007', 'val'), ('2012', 'train'), ('2012', 'val'))
    # pascal_data = load_pascal_dataset()
    # centroids = pascal_data[np.random.choice(np.arange(len(pascal_data)), k, replace=False)]
    # # centroids = pascal_data[:k]
    # pascal_anchors = kmeans_iou(k, centroids, pascal_data, feature_size=img_size / 32)

    # datasets = ('train2014', 'val2014')
    # # datasets = ('test2014', 'test2015')
    # coco_data = load_coco_dataset()
    # centroids = coco_data[np.random.choice(np.arange(len(coco_data)), k, replace=False)]
    # # centroids = coco_data[:k]
    # coco_anchors = kmeans_iou(k, centroids, coco_data, feature_size=img_size / 32)

    # reshape: [[x1,y1,w1,h1],...,[xn,yn,wn,hn]]
    # pascal_anchors = np.hstack((np.zeros((k, 2)), pascal_anchors * img_size))
    # coco_anchors = np.hstack((np.zeros((k, 2)), coco_anchors * img_size))

    # # Hardcoded results to skip computation
    # pascal_anchors = np.asarray(
    # [[0.,            0.,          295.43055556,  157.86944444],
    #  [0.,            0.,           45.14861187,   62.17800762],
    #  [0.,            0.,          361.03531786,  323.51160444],
    #  [0.,            0.,          160.96848934,  267.35032437],
    #  [0.,            0.,          103.46714456,  131.91278375]])
    #
    # coco_anchors = np.asarray(
    # [[0.,            0.,          159.01092483,  118.88467982],
    #  [0.,            0.,           64.93587645,   58.59559227],
    #  [0.,            0.,          475.61739025,  332.0915918 ],
    #  [0.,            0.,          195.32259936,  272.98913297],
    #  [0.,            0.,           19.46456615,   21.47707798]])
    # pascal_anchors = pascal_anchors[pascal_anchors[:, 2].argsort()]
    # coco_anchors = coco_anchors[coco_anchors[:, 2].argsort()]

    # Hardcode anchor center coordinates for plot
    # pascal_anchors[3][0] = 250
    # pascal_anchors[3][1] = 100
    # pascal_anchors[0][0] = 300
    # pascal_anchors[0][1] = 450
    # pascal_anchors[4][0] = 650
    # pascal_anchors[4][1] = 250
    # pascal_anchors[2][0] = 110
    # pascal_anchors[2][1] = 350
    # pascal_anchors[1][0] = 300
    # pascal_anchors[1][1] = 300

    # # Reorder centroid 2 and 5 in coco anchors
    # tmp = np.copy(coco_anchors[2])
    # coco_anchors[2] = coco_anchors[3]
    # coco_anchors[3] = tmp

    # coco_anchors[3][0] = 250
    # coco_anchors[3][1] = 100
    # coco_anchors[0][0] = 300
    # coco_anchors[0][1] = 450
    # coco_anchors[4][0] = 650
    # coco_anchors[4][1] = 250
    # coco_anchors[2][0] = 110
    # coco_anchors[2][1] = 350
    # coco_anchors[1][0] = 300
    # coco_anchors[1][1] = 300

    # plot_anchors(pascal_anchors, coco_anchors)

    print('done')
