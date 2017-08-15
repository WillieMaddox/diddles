from operator import itemgetter
import IO

import networkx as nx
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn

# def traverse(graph, start, node):
#     graph.depth[node.name] = node.shortest_path_distance(start)
#     for child in node.hyponyms():
#         graph.add_edge(node.name, child.name)
#         traverse(graph, start, child)
#
# def hyponym_graph(start):
#     G = nx.Graph()
#     G.depth = {}
#     traverse(G, start, start)
#     return G
#
# def graph_draw(graph):
#     nx.draw_graphviz(graph,
#         node_size=[16 * graph.degree(n) for n in graph],
#         node_color=[graph.depth[n] for n in graph],
#         with_labels=False)
#     plt.show()
#
# dog = wn.synset('dog.n.01')
# graph = hyponym_graph(dog)
# nx.draw(graph)
# plt.show()

IO.remove_orphaned_images()
IO.remove_orphaned_metadata()

# *** Custom words ***

dog_breeds_file = 'dog_breeds.names'
with open(dog_breeds_file) as ifs:
    dog_breeds = set(ifs.read().strip().lower().split('\n'))
print('{:13s}:{:10d}'.format('# dog_breeds', len(dog_breeds)))

cat_breeds_file = 'cat_breeds.names'
with open(cat_breeds_file) as ifs:
    cat_breeds = set(ifs.read().strip().lower().split('\n'))
print('{:13s}:{:10d}'.format('# cat_breeds', len(cat_breeds)))

car_makes_file = 'car_makes.names'
with open(car_makes_file) as ifs:
    car_makes = set(ifs.read().strip().lower().split('\n'))
print('{:13s}:{:10d}'.format('# car_makes', len(car_makes)))

# *** Pixabay files ***

pixabay_labels = IO.read_pixabay_metadata_file()
print('{:13s}:{:10d}'.format('# labels', len(pixabay_labels)))

pixabay_blacklist = IO.read_pixabay_blacklist_file()
print('{:13s}:{:10d}'.format('# blacklist', len(pixabay_blacklist)))

pixabay_aliases = IO.read_pixabay_aliases_file()
print('{:13s}:{:10d}'.format('# aliases', len(pixabay_aliases)))

pixabay_tallies = IO.read_pixabay_tally_file()
print('{:13s}:{:10d}'.format('# tallies', len(pixabay_tallies)))

pwords_set = set(pixabay_tallies.keys())
c_total = len(pwords_set)
c_tally_total = sum(pixabay_tallies.values())
assert len(pwords_set) == len(pixabay_tallies)
print('{:13s}:{:10d}{:10d}'.format('ptallies', c_total, c_tally_total))

# *** Imagenet files ***

# DAG = IO.read_wordnet_heirarchy_file()
# print nx.ancestors(DG, 'n02691156')

synset_words_dict = IO.read_synset_words_file()

print('\n{:>13s}:{:>10s}{:>10s}{:>13s}'.format('', 'list', 'set', 'intersection'))
swords = [s for ss in synset_words_dict.itervalues() for s in ss]
swords_set = set(swords)
words_set = swords_set.intersection(pwords_set)
print('{:13s}:{:10d}{:10d}{:13d}'.format('default', len(swords), len(swords_set), len(words_set)))
# lowercase
swords_lower = [s.lower() for s in swords]
swords_lower_set = set(swords_lower)
words_lower_set = swords_lower_set.intersection(pwords_set)
print('{:13s}:{:10d}{:10d}{:13d}'.format('lower', len(swords_lower), len(swords_lower_set), len(words_lower_set)))

# *** Wordnet files ***

wordnet_nouns = set(IO.read_wordnet_index_file("noun"))
wordnet_nouns = IO.read_wordnet_exc_file("noun", output=wordnet_nouns)
wordnet_verbs = set(IO.read_wordnet_index_file("verb"))
wordnet_verbs = IO.read_wordnet_exc_file("verb", output=wordnet_verbs)
wordnet_adv = set(IO.read_wordnet_index_file("adv"))
wordnet_adv = IO.read_wordnet_exc_file("adv", output=wordnet_adv)
wordnet_adj = set(IO.read_wordnet_index_file("adj"))
wordnet_adj = IO.read_wordnet_exc_file("adj", output=wordnet_adj)

# ii = 0
pixabay_labels_new = {}
for key, metadata in pixabay_labels.iteritems():
    metadata_new = {}
    labels_new = metadata['tags']
    # if 'pelikan' in labels_new:
    #     if 'pelican' in labels_new:
    #         ii += 1
    #     else:
    #         print labels_new

    if len(labels_new) > 0:
        labels_new = labels_new.difference(pixabay_blacklist)

    if len(labels_new) > 0:
        labels_new = labels_new.difference(wordnet_nouns)

    if len(labels_new) > 0:
        labels_new = labels_new.difference(wordnet_verbs)

    if len(labels_new) > 0:
        labels_new = labels_new.difference(wordnet_adj)

    if len(labels_new) > 0:
        labels_new = labels_new.difference(wordnet_adv)

    if len(labels_new) > 0:
        labels_new = labels_new.difference(dog_breeds)

    if len(labels_new) > 0:
        labels_new = labels_new.difference(cat_breeds)

    if len(labels_new) > 0:
        labels_new = labels_new.difference(car_makes)

    for _, alias_list in pixabay_aliases.iteritems():
        if len(labels_new) > 0:
            labels_new = labels_new.difference(alias_list)

    labels2 = set()
    for pword in labels_new:
        if pword[-1] == 's' and pword[:-1] in labels_new:
            continue

        if pword[-1] == 's' and pword[:-1] in words_set:
            continue

        elif pword[-2:] == 'es' and pword[:-2] in words_set:
            continue

        elif pword[-3:] == 'ies' and pword[:-3] + 'y' in words_set:
            continue

        elif pword[-3:] == 'ing' and pword[:-3] in words_set:
            continue

        elif pword[-1] == 's' and pword[:-1] in words_lower_set:
            continue

        elif pword[-2:] == 'es' and pword[:-2] in words_lower_set:
            continue

        elif pword[-3:] == 'ies' and pword[:-3] + 'y' in words_lower_set:
            continue

        elif pword[-3:] == 'ing' and pword[:-3] in words_lower_set:
            continue

        elif ' ' in pword and pword.replace(' ', '') in words_set:
            continue

        elif ' ' in pword and pword.replace(' ', '') in words_lower_set:
            continue

        labels2.add(pword)
    labels_new = labels2

    metadata_new['tags'] = list(labels_new)
    pixabay_labels_new[key] = metadata_new

# print 'found:', ii
max_tag_set = 0
for value in pixabay_labels.itervalues():
    max_tag_set = max(max_tag_set, len(value['tags']))
max_tag_set += 1

tag_counts_old = {i: 0 for i in range(max_tag_set)}
for value in pixabay_labels.itervalues():
    tag_counts_old[len(value['tags'])] += 1

set_a = set()
tag_counts_new = {i: 0 for i in range(max_tag_set)}
for key, value in pixabay_labels_new.iteritems():
    if len(value['tags']) > 7:
        print(key, value)
    tag_counts_new[len(value['tags'])] += 1
    if len(value['tags']) > 0:
        set_a.update(value['tags'])

print(len(set_a))
total_tag_counts_old = sum(tag_counts_old.values())
total_tag_counts_new = sum(tag_counts_new.values())
for i in range(max_tag_set):
    if tag_counts_old[i] == 0 and tag_counts_new[i] == 0:
        continue
    print("{:3d} {:6d} {:6d} {:10.3f} {:10.3f}".format(i, tag_counts_old[i], tag_counts_new[i], 100.0*tag_counts_old[i]/total_tag_counts_old, 100.0*tag_counts_new[i]/total_tag_counts_new))
print("All {:6d} {:6d}".format(total_tag_counts_old, total_tag_counts_new))

set_b = set()
blacklist_new = {}
key_list = ['0', 'dogs', 'cats', 'cars', 'nouns', 'verbs', 'adv', 'adj', 'blacklist', 'alias', 'ws']
counts = {key: 0 for key in key_list}
counts_tally = {key: 0 for key in key_list}

print_delims = {' ': '_', '-': '-'}
delims = [(' ', '', '_1'), (' ', '-', '_2'), ('-', ' ', '_3'), ('-', '', '_4')]
suffi = [('s', ''), ('es', ''), ('ies', 'y'), ('ing', ''), ('ed', '')]

for pword, ptally in pixabay_tallies.iteritems():
    key = '0'

    # if pword == "coca-cola":
    #     print pword

    if pword in dog_breeds:
        key = 'dogs'

    elif pword in cat_breeds:
        key = 'cats'

    elif pword in car_makes:
        key = 'cars'

    elif pword in wordnet_nouns:
        key = 'nouns'

    elif pword in wordnet_verbs:
        key = 'verbs'

    elif pword in wordnet_adv:
        key = 'adv'

    elif pword in wordnet_adj:
        key = 'adj'

    elif pword in pixabay_blacklist:
        key = 'blacklist'

    elif pword in words_set:
        key = 'ws'

    elif pword in words_lower_set:
        key = 'wsl'

    else:
        for _, alias_list in pixabay_aliases.iteritems():
            if pword in alias_list:
                key = 'alias'
                break

    # if iwc == 1:
    #     key = 'alias'
    #     continue

    if key != '0':
        counts[key] += 1
        counts_tally[key] += ptally
        continue
    else:
        # candidates = {'': pword}
        candidates = {}

        dds = []
        for dd, delim in enumerate(delims):
            if delim[0] in pword:
                dds.append(dd)
                sdkey = delim[2]
                candidates[sdkey] = pword.replace(delim[0], delim[1])
        sss = []
        for ss, suffix in enumerate(suffi):
            if pword[-len(suffix[0]):] == suffix[0]:
                sss.append(ss)
                candidates[':' + suffix[0]] = pword[:-len(suffix[0])] + suffix[1]

        for dd in dds:
            delim = delims[dd]
            for ss in sss:
                suffix = suffi[ss]
                sdkey = delim[2] + suffix[0]
                candidate = pword.replace(delim[0], delim[1])
                candidates[sdkey] = candidate[:-len(suffix[0])] + suffix[1]

    ws_cv = words_set.intersection(set(candidates.values()))
    if len(ws_cv) > 0:
        if len(ws_cv) > 1:
            print(pword, ws_cv)
        for dkey, word in candidates.iteritems():
            if word in words_set:
                key = 'ws' + dkey

    if key not in key_list:
        key_list.append(key)
        counts[key] = 0
        counts_tally[key] = 0

    if key != '0':
        counts[key] += 1
        counts_tally[key] += ptally
        continue

    wsl_cv = words_lower_set.intersection(set(candidates.values()))
    if len(wsl_cv) > 0:
        if len(wsl_cv) > 1:
            print(pword, wsl_cv)
        for dkey, word in candidates.iteritems():
            if word in words_lower_set:
                key = 'wsl' + dkey

    if key not in key_list:
        key_list.append(key)
        counts[key] = 0
        counts_tally[key] = 0

    counts[key] += 1
    counts_tally[key] += ptally

    if key is '0':
        blacklist_new[pword] = ptally
        set_b.add(pword)

with open('pixabay_blacklist00.txt', 'w') as ofs:
    for bl, c in sorted(blacklist_new.items(), key=itemgetter(1), reverse=True):
        ofs.write("{}\t{}\n".format(c, bl))

print(set_a.symmetric_difference(set_b))

c_sum = sum(counts.values())
c_tally_sum = sum(counts_tally.values())

fmt = '{:13s}:{:10d}{:10d}{:10.3f}{:10.3f}'
for k in key_list:
    if counts[k] == 0 and counts_tally[k] == 0:
        continue
    print(fmt.format(k, counts[k], counts_tally[k], 100.0 * counts[k] / c_total, 100.0 * counts_tally[k] / c_tally_total))

print(fmt.format('sum', c_sum, c_tally_sum, 100.0 * c_sum / c_total, 100.0 * c_tally_sum / c_tally_total))

