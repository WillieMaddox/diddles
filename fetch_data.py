#!/usr/bin/env python
"""
Get data from pixabay. download images to dirname, and write
Caffe ImagesDataLayer training file.
"""
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from multiprocessing.pool import ThreadPool
from operator import itemgetter

import requests
from PIL import Image
from skimage import io

import IO
import utils
from api_keys import PIXABAY_API_KEY

PER_PAGE = 200
PULL_PERCENTAGE = 0.9


class UserCredit(object):
    def __init__(self):
        self.max_credit = 4500
        self.duration = 7200
        self.timestamps = []

    def __call__(self):
        now = time.time()
        while True:
            if len(self.timestamps) == 0:
                break
            if now - self.timestamps[0] > self.duration:
                # purge expired timestamps
                self.timestamps = self.timestamps[1:]
                continue
            if len(self.timestamps) > self.max_credit:
                # delay all execution until more credit is available.
                time.sleep(10)
                now = time.time()
                continue
            break
        self.timestamps.append(now)

    def __str__(self):
        return f"{self.max_credit - len(self.timestamps)}"

    @staticmethod
    def sleep(n):
        # forcefully suspend all execution for n seconds.
        time.sleep(n)

UC = UserCredit()


def download_metadata(labels, page=1, per_page=PER_PAGE):
    """

    Returns
    -------
    A python dict / json
    """
    base_url = 'https://pixabay.com/api/'
    query = {'key': PIXABAY_API_KEY,
             'q': '+'.join(labels),
             'image_type': 'photo',
             # 'order': 'latest',
             'per_page': str(per_page),
             'page': str(page)}
    query_list = ['='.join([k, v]) for k, v in query.items()]
    query_string = '&'.join(query_list)
    url = '?'.join([base_url, query_string])
    response = requests.get(url, headers={'content-type': 'application/json'})
    UC()
    return response.json() if response.status_code == 200 else response.content


def download_and_save_image(args_tuple):
    idx, url, filename = args_tuple
    try:
        if filename.endswith('.jpg'):
            if not os.path.exists(filename):
                urllib.request.urlretrieve(url, filename)
                _ = io.imread(filename)
        elif filename.endswith('.png'):
            new_filename = ''.join([filename[:-3], 'jpg'])
            if not os.path.exists(new_filename):
                urllib.request.urlretrieve(url, filename)
                im = Image.open(filename)
                im.save(new_filename, "JPEG")
                os.remove(filename)
        UC()
        return idx, True
    except Exception as e:
        print('Bad url or image')
        print(e)
        if os.path.exists(filename):
            print(f'Deleting file {filename}...', end=' ')
            os.remove(filename)
            print('Deleted')
        return idx, False


def filter_labels(old_tags):
    new_tags = set()
    for tag in list(old_tags):
        if tag.startswith('the '):
            tag = tag[4:]
        if tag in PIXABAY_ALIASES:
            tag = PIXABAY_ALIASES[tag]

        if tag in BLACKLIST or tag in BLACKLIST_LOWER:
            continue
        elif tag[-1] == 's' and tag[:-1] in BLACKLIST:
            continue
        elif tag[-1] == 's' and tag[:-1] in BLACKLIST_LOWER:
            continue
        elif tag[-2:] == 'es' and tag[:-2] in BLACKLIST:
            continue
        elif tag[-3:] == 'ies' and tag[:-3] + 'y' in BLACKLIST:
            continue
        elif tag[-3:] == 'ing' and tag[:-3] in BLACKLIST:
            continue
        elif tag in WHITELIST or tag in WHITELIST_LOWER:
            new_tags.add(tag)
            continue
        elif tag[-1] == 's' and tag[:-1] in WHITELIST:
            continue
        elif tag[-1] == 's' and tag[:-1] in WHITELIST_LOWER:
            continue
        elif tag[-2:] == 'es' and tag[:-2] in WHITELIST:
            continue
        elif tag[-3:] == 'ies' and tag[:-3] + 'y' in WHITELIST:
            continue
        elif tag[-3:] == 'ing' and tag[:-3] in WHITELIST:
            continue
        else:
            new_tags.add(tag)

    return new_tags


def update_used_labels(curr_labels, image_ids):

    for used_label_group, old_image_ids in iter(USED_LABELS.items()):
        if used_label_group.issubset(curr_labels):
            USED_LABELS[used_label_group].update(image_ids)
        # if used_label_group.issuperset(curr_labels):
        #     image_ids.update(USED_LABELS[used_label_group])

    USED_LABELS[frozenset(curr_labels)] = image_ids


def get_image_metadata(meta, curr_labels):

    total = 0
    n_hits = 0
    n_new = 0
    n_updated = 0
    page = 0
    image_ids = set()

    while page <= 3:
        page += 1
        temp = download_metadata(curr_labels, page=page)

        if isinstance(temp, str):
            print(temp)
            print('suspending for 60 seconds...')
            UC.sleep(60)
            break

        if temp['total'] == 0:
            print("temp['total'] == 0", "curr_labels:", curr_labels, "page:", page)
            print('suspending for 60 seconds...')
            UC.sleep(60)
            break

        total = temp['total']
        for record in temp['hits']:
            n_hits += 1

            grouped_tags = set([tag.strip(' ') for tag in record['tags'].split(',')])
            grouped_tags.update(curr_labels)
            if record['id'] in meta:
                orig_tags = meta[record['id']]['tags']
                grouped_tags.update(orig_tags)
            else:
                orig_tags = []

            new_tags = filter_labels(grouped_tags)
            image_ids.add(record['id'])

            if len(orig_tags) == 0:
                n_new += 1
            elif len(orig_tags) != len(new_tags):
                n_updated += 1
            else:
                continue

            meta[record['id']] = {'tags': new_tags,
                                  'height': record['webformatHeight'],
                                  'width': record['webformatWidth'],
                                  'webformatURL': record['webformatURL']}

        if temp['totalHits'] <= page * PER_PAGE:
            break

    update_used_labels(curr_labels, image_ids)

    assert len(USED_LABELS[frozenset(curr_labels)]) == n_hits
    # print out some logging info.
    print(f'{len(meta):7d} '
          f'{total:6d} '
          f'{total * PULL_PERCENTAGE:6.0f} '
          f'{n_hits:6d} '
          f'{n_updated:5d} '
          f'{n_new:5d}', frozenset(curr_labels))

    return meta, temp['total']


def get_new_label_set(new_image_metadata, curr_labels):

    label_counts = utils.get_counts(new_image_metadata)

    for label_name, _ in sorted(label_counts.items(), key=itemgetter(1), reverse=True):

        if label_name in curr_labels:
            continue

        new_labels = curr_labels.union([label_name])

        if len('+'.join(new_labels)) > 100:
            # pixabay api rule
            continue
        dup_set_found = False
        for used_label_group, image_ids in iter(USED_LABELS.items()):
            if len(used_label_group.symmetric_difference(new_labels)) == 0:
                dup_set_found = True
                break

        if not dup_set_found:
            break
    else:
        new_labels = set()

    return new_labels


def create_image_metadata_subset(curr_metadata, curr_labels):
    new_image_metadata = {}
    for idx, meta in iter(curr_metadata.items()):
        if curr_labels.issubset(meta['tags']):
            new_image_metadata[idx] = meta
    return new_image_metadata


def merge_image_metadata_sets(current_metadata, future_metadata):
    for idx, meta in iter(future_metadata.items()):
        current_metadata[idx] = meta
    return current_metadata


def recurse_labels(current_metadata, current_labels):

    current_metadata, total = get_image_metadata(current_metadata, current_labels)

    while True:

        if len(USED_LABELS[frozenset(current_labels)]) > total * PULL_PERCENTAGE:
            break
        if len(current_metadata) > total * PULL_PERCENTAGE:
            break

        new_image_metadata = create_image_metadata_subset(current_metadata, current_labels)
        new_labels = get_new_label_set(new_image_metadata, current_labels)
        new_image_metadata = recurse_labels(new_image_metadata, new_labels)
        current_metadata = merge_image_metadata_sets(current_metadata, new_image_metadata)

    return current_metadata


PIXABAY_ALIASES = IO.read_pixabay_aliases_file(dual=True)

WHITELIST = IO.get_whitelist_words()
WHITELIST_LOWER = set([w.lower() for w in list(WHITELIST)])

wordnet_nouns = IO.read_wordnet_exc_file("noun")
BLACKLIST = set(wordnet_nouns.keys())
BLACKLIST = IO.read_wordnet_index_file("verb", output=BLACKLIST)
BLACKLIST = IO.read_wordnet_exc_file("verb", output=BLACKLIST)
BLACKLIST = IO.read_wordnet_index_file("adv", output=BLACKLIST)
BLACKLIST = IO.read_wordnet_exc_file("adv", output=BLACKLIST)
BLACKLIST = IO.read_wordnet_index_file("adj", output=BLACKLIST)
BLACKLIST = IO.read_wordnet_exc_file("adj", output=BLACKLIST)
BLACKLIST.difference_update(WHITELIST)
BLACKLIST_LOWER = set([w.lower() for w in list(BLACKLIST)])

USED_LABELS = {}

if __name__ == '__main__':

    with open('data/yolo_diddles.names') as ifs:
        labels = ifs.read().strip().split('\n')

    IO.merge_orphaned_metadata()
    IO.merge_orphaned_images()

    labelsdata = IO.read_pixabay_metadata_file()

    # new_labelsdata = {}
    # for idx, meta in iter(labelsdata.items()):
    #     new_labelsdata[idx] = {}
    #     new_tags = filter_labels(meta['tags'])
    #     new_labelsdata[idx]['tags'] = new_tags
    #     new_labelsdata[idx]['width'] = meta['width']
    #     new_labelsdata[idx]['height'] = meta['height']
    #     IO.write_pixabay_metadata_file(new_labelsdata)
    # label_counts_new = utils.get_counts(new_labelsdata)
    # IO.write_pixabay_tally_file(label_counts_new)

    for label in labels:

        image_meta = {idx: meta for idx, meta in labelsdata.items() if label in meta['tags']}

        print(' -- Beginning:', label, '--')
        print(f'{"Current":>7s} '
              f'{"total":>6s} '
              f'{"frac":>6s} '
              f'{"hits":>6s} '
              f'{"upd":>5s} '
              f'{"new":>5s}')

        image_meta = recurse_labels(image_meta, {label})

        url_filename_list = []
        labels_updated = False
        for idx, record in iter(image_meta.items()):
            if idx in labelsdata:
                if labelsdata[idx]['width'] != record['width']:
                    if labelsdata[idx]['height'] != record['height']:
                        print('\nIMAGE CHANGED ON PIXABAY!!!')
                        print(f'record: {idx}')
                        print(f'old: {labelsdata}')
                        print(f'new: {record}')
                        print('\n')
                        continue

                # update existing tags
                tags_old = labelsdata[idx]['tags']
                tags_new = record['tags']
                if len(tags_old.intersection(tags_new)) != len(tags_new):
                    tags_new.update(tags_old)
                    labelsdata[idx]['tags'] = tags_new
                    labels_updated = True

            else:
                # Create the list of image files to download
                url = record['webformatURL']
                filetype = url.split('.')[-1]
                filename = f"{IO.pixabay_image_dir}/{idx}.{filetype}"
                url_filename_list.append((idx, url, filename))

        # download new images in parallel
        start = time.time()
        n_records = len(url_filename_list)
        print(f"Remain: {n_records}, Credit: {UC}, Timer: {time.time() - start}")
        # don't set pooling too high.  remember you can only download 5000 images per hour.
        results = ThreadPool(2).imap_unordered(download_and_save_image, url_filename_list)
        for idx, result in results:
            if result:
                labelsdata[idx] = {}
                labelsdata[idx]['tags'] = image_meta[idx]['tags']
                labelsdata[idx]['width'] = image_meta[idx]['width']
                labelsdata[idx]['height'] = image_meta[idx]['height']
                labels_updated = True
                n_records -= 1
                if labels_updated and n_records % 100 == 0:
                    print(f"Remain: {n_records}, Credit: {UC}, Timer: {time.time() - start}")
            else:
                UC.sleep(60)

        # update the labels file with the new and updated labels.
        if labels_updated:
            IO.update_files(labelsdata)
        print(f'label: {label} completed')

    IO.remove_orphaned_images()

    print('done')
