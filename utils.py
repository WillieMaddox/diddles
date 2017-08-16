
def to_bytestring(s, enc='utf-8'):
    """Convert the given unicode string to a bytestring, using the standard encoding,
    unless it's already a bytestring"""
    return s.encode(enc) if not isinstance(s, str) else s


def safe_unicode(obj, *args):
    """ return the unicode representation of obj """
    try:
        return str(obj, *args)
    except UnicodeDecodeError:
        # obj is byte string
        ascii_text = str(obj).encode('string_escape')
        return str(ascii_text)


def safe_bytestring(obj):
    """ return the byte string representation of obj """
    try:
        return str(obj)
    except UnicodeEncodeError:
        # obj is unicode
        return str(obj).encode('unicode_escape')


def decode_list(data):
    rv = []
    for item in data:
        if isinstance(item, str):
            item = item.encode('utf-8')
        elif isinstance(item, list):
            item = decode_list(item)
        elif isinstance(item, dict):
            item = decode_dict(item)
        rv.append(item)
    return rv


def decode_dict(data):
    rv = {}
    for key, value in iter(data.items()):
        if isinstance(key, str):
            key = key.encode('utf-8')
        if isinstance(value, str):
            value = value.encode('utf-8')
        elif isinstance(value, list):
            value = decode_list(value)
        elif isinstance(value, dict):
            value = decode_dict(value)
        rv[key] = value
    return rv


def iou(a, b):
    return len(a.intersection(b))/len(a.union(b))


class BBoxDim(object):

    def __init__(self):
        self._min = None
        self._max = None
        self._cen = None
        self._thick = None
        self.n_inputs = 0

    def reset(self):
        self.__init__()

    def update(self):
        if self.n_inputs != 2:
            return
        if self._min is not None and self._max is not None:
            self._cen = (self._max + self._min) / 2
            self._thick = self._max - self._min
        elif self._min is not None and self._cen is not None:
            self._max = 2 * self._cen - self._min
            self._thick = 2 * (self._cen - self._min)
        elif self._min is not None and self._thick is not None:
            self._max = self._min + self._thick
            self._cen = self._min + self._thick / 2
        elif self._max is not None and self._cen is not None:
            self._min = 2 * self._cen - self._max
            self._thick = 2 * (self._max - self._cen)
        elif self._max is not None and self._thick is not None:
            self._min = self._max - self._thick
            self._cen = self._max - self._thick / 2
        elif self._cen is not None and self._thick is not None:
            self._min = self._cen - self._thick / 2
            self._max = self._cen + self._thick / 2
        self.n_inputs = 4

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, val):
        if self._min is None:
            self.n_inputs += 1
            self._min = val
            self.update()

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, val):
        if self._max is None:
            self.n_inputs += 1
            self._max = val
            self.update()

    @property
    def cen(self):
        return self._cen

    @cen.setter
    def cen(self, val):
        if self._cen is None:
            self.n_inputs += 1
            self._cen = val
            self.update()

    @property
    def thick(self):
        return self._thick

    @thick.setter
    def thick(self, val):
        if self._thick is None:
            self.n_inputs += 1
            self._thick = val
            self.update()


class BoundingBox(object):
    """
    size is a 2 element list
    bbox is a dictionary
    fmt is the format of the bbox. (i.e. voc, coco, fgvc, darknet
    """
    def __init__(self, size, bbox, fmt):
        self.fmt = fmt
        self.dw = 1. / size[0]
        self.dh = 1. / size[1]
        self.bbox_orig = bbox

        self.bbox_formats = {'voc': ('xmin', 'xmax', 'ymin', 'ymax'),
                             'fgvc': ('xmin', 'ymin', 'xmax', 'ymax'),
                             'coco': ('xmin', 'ymin', 'xthick', 'ythick'),
                             'darknet': ('xcen', 'ycen', 'xthick', 'ythick')}

        self.x = BBoxDim()
        self.y = BBoxDim()

        for key, val in zip(self.bbox_formats[fmt], bbox):
            if key[0] in ('x', 'X'):
                if fmt == 'darknet':
                    val = int(round(val / self.dw))
                if key == 'xmin':
                    self.x.min = val
                elif key == 'xmax':
                    self.x.max = val
                elif key == 'xcen':
                    self.x.cen = val
                elif key == 'xthick':
                    self.x.thick = val
            if key[0] in ('y', 'Y'):
                if fmt == 'darknet':
                    val = int(round(val / self.dh))
                if key == 'ymin':
                    self.y.min = val
                elif key == 'ymax':
                    self.y.max = val
                elif key == 'ycen':
                    self.y.cen = val
                elif key == 'ythick':
                    self.y.thick = val

    def convert_to(self, fmt):

        bbox = []
        for key in self.bbox_formats[fmt]:
            val = 0
            if key[0] in ('x', 'X'):
                if key == 'xmin':
                    val = self.x.min
                elif key == 'xmax':
                    val = self.x.max
                elif key == 'xcen':
                    val = self.x.cen
                elif key == 'xthick':
                    val = self.x.thick
                else:
                    raise
                if fmt == 'darknet':
                    val = val * self.dw
            if key[0] in ('y', 'Y'):
                if key == 'ymin':
                    val = self.y.min
                elif key == 'ymax':
                    val = self.y.max
                elif key == 'ycen':
                    val = self.y.cen
                elif key == 'ythick':
                    val = self.y.thick
                else:
                    raise
                if fmt == 'darknet':
                    val = val * self.dh
            bbox.append(val)
        return bbox


def convert_coco_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def get_yolo_filters(classes, coords, num):
    return (classes + coords + 1) * num


def print_class_counts(name, class_counts, skip_zeros=False, mapper=None):
    print('--------{}--------'.format(name))
    for label, count in iter(class_counts.items()):
        if skip_zeros and count == 0:
            continue
        lbl = mapper[label] if mapper else label
        print('{:15s} {:10d}'.format(lbl, count))
    print('------------------------')


def check_word(word, words_set, suffix=None):
    if suffix:
        if word.endswith(suffix):
            test_word = word.rstrip(suffix)
    else:
        test_word = word

    iword = words_set.intersection({test_word})
    if len(iword) == 0:
        return 0
    elif len(iword) == 1:
        return 1
    elif len(iword) > 1:
        return 2


def get_counts(md):
    lc = {}
    for record in iter(md.values()):
        for tag in record['tags']:
            if tag not in lc:
                lc[tag] = 0
            lc[tag] += 1
    return lc


def simplify_json(meta_old, keys_to_keep):
    meta_new = {}
    for name, record_old in iter(meta_old.items()):
        record_new = {}
        for key, value in iter(record_old.items()):
            if key in keys_to_keep:
                record_new[key] = value
        meta_new[name] = record_new
    return meta_new


def extract_tags(meta_old):
    meta_new = {}
    for idx, record_old in iter(meta_old.items()):
        meta_new[int(idx)] = record_old['tags']
    return meta_new


