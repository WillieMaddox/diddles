from operator import itemgetter
import IO
from utils import safe_unicode
import enchant
import unicodedata
import nltk
nltk.data.path.append("/media/Borg_LS/DATA/nltk_data")
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from nltk.stem import RegexpStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer

from replacers import RepeatReplacer
from replacers import RegexpReplacer
from replacers import SpellingReplacer
from replacers import YamlWordReplacer

import networkx as nx
import matplotlib.pyplot as plt

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

dog_breeds_file = 'data/dog_breeds.names'
with open(dog_breeds_file) as ifs:
    dog_breeds = set(ifs.read().strip().lower().split('\n'))
print('{:13s}:{:10d}'.format('# dog_breeds', len(dog_breeds)))

hypo = lambda s: s.hyponyms()

syn0 = wn.synset('dog.n.01')
syn_words = list(syn0.closure(hypo))
i = 0
j = 0
for breed_str in sorted(dog_breeds):
    found = False
    breed = safe_unicode(breed_str)
    b = breed.replace(' ', '_') if ' ' in breed else breed
    for syn in wn.synsets(b):
        if found:
            break
        for syn_path in syn.hypernym_paths():
            hyper_names = [s.name() for s in syn_path]
            if 'dog.n.01' in hyper_names:
                if syn in syn_words:
                    syn_words.pop(syn_words.index(syn))
                    found = True
                    break
    if found:
        i += 1
    else:
        j += 1
        # print j, b

for j, syn in enumerate(sorted(syn_words)):
    print(j, syn.name())

print(i, j)

cat_breeds_file = 'data/cat_breeds.names'
with open(cat_breeds_file) as ifs:
    cat_breeds = set(ifs.read().strip().lower().split('\n'))
print('{:13s}:{:10d}'.format('# cat_breeds', len(cat_breeds)))

syn0 = wn.synset('feline.n.01')
syn_words = list(syn0.closure(hypo))
i = 0
j = 0
for breed_str in sorted(cat_breeds):
    found = False
    breed = safe_unicode(breed_str)
    b = breed.replace(' ', '_') if ' ' in breed else breed
    for syn in wn.synsets(b):
        if found:
            break
        for syn_path in syn.hypernym_paths():
            hyper_names = [s.name() for s in syn_path]
            if 'feline.n.01' in hyper_names:
                if syn in syn_words:
                    syn_words.pop(syn_words.index(syn))
                    found = True
                    break
    if found:
        i += 1
    else:
        j += 1
        # print j, b

for j, syn in enumerate(sorted(syn_words)):
    print(j, syn.name())

print(i, j)

car_makes_file = 'data/car_makes.names'
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

replacer = SpellingReplacer()
stemmerporter = PorterStemmer()
stemmerlan = LancasterStemmer()
lemmatizer = WordNetLemmatizer()
yamlreplacer = YamlWordReplacer('data/synonyms.yaml')

# good0 = 0
# fixed0 = 0
# skipped = 0
# for pword, ptally in pixabay_tallies.iteritems():
#     if ptally <= 2:
#         skipped += 1
#         continue
#     pword_uni = safe_unicode(pword)
#     # pword_uni = unicodedata.normalize('NFKD', pword_uni).encode('ascii', 'ignore')
#     if ' ' in pword_uni:
#         pwords_uni = pword_uni.split()
#     else:
#         pwords_uni = [pword_uni]
#
#     fixed = False
#     pwords_new = []
#     for pw in pwords_uni:
#         pw_new = replacer.replace(pw)
#
#         if pw == pw_new:
#             pwords_new.append(pw)
#         else:
#             pwords_new.append(pw_new)
#             fixed = True
#
#     if fixed:
#         fixed0 += 1
#     else:
#         good0 += 1
#
# print 'good:', good0
# print 'fixed:', fixed0
# print 'skipped:', skipped

n_hits = 2

pixabay_labels_newA = {}
flagged_words = set()
for key, metadata in pixabay_labels.items():
    metadata_new = {}
    labels_new = set()
    pwords_uni = set([safe_unicode(pword) for pword in metadata['tags']])
    for pword_uni in pwords_uni:
        if pword_uni in flagged_words:
            continue
        if pixabay_tallies[pword_uni] <= n_hits:
            flagged_words.update([pword_uni])
            continue
        synsets = wn.synsets(pword_uni)
        if len(synsets) > 0:
            flagged_words.update([pword_uni])
            continue
        cword_uni = pword_uni.replace(' ', '_') if ' ' in pword_uni else pword_uni
        cword_uni = yamlreplacer.replace(cword_uni)

        synsets = wn.synsets(cword_uni)
        if len(synsets) > 0:
            flagged_words.update([pword_uni])
            continue
        # pword_new = replacer.replace(pword_uni)
        # synsets = wn.synsets(pword_new)
        # if len(synsets) > 0:
        #     flagged_words.update([pword_uni])
        #     continue

        labels_new.update([pword_uni])

    # if len(labels_new) > 0:
    #     labels_new = labels_new.difference(pixabay_blacklist)
    #
    # if len(labels_new) > 0:
    #     labels_new = labels_new.difference(dog_breeds)
    #
    # if len(labels_new) > 0:
    #     labels_new = labels_new.difference(cat_breeds)
    #
    # if len(labels_new) > 0:
    #     labels_new = labels_new.difference(car_makes)
    #
    # for _, alias_list in pixabay_aliases.iteritems():
    #     if len(labels_new) > 0:
    #         labels_new = labels_new.difference(alias_list)

    # labels2 = set()
    # for pword in labels_new:
    #     if pword[-1] == 's' and pword[:-1] in words_set:
    #         continue
    #
    #     elif pword[-2:] == 'es' and pword[:-2] in words_set:
    #         continue
    #
    #     elif pword[-3:] == 'ies' and pword[:-3] + 'y' in words_set:
    #         continue
    #
    #     elif pword[-3:] == 'ing' and pword[:-3] in words_set:
    #         continue
    #
    #     elif pword[-1] == 's' and pword[:-1] in words_lower_set:
    #         continue
    #
    #     elif pword[-2:] == 'es' and pword[:-2] in words_lower_set:
    #         continue
    #
    #     elif pword[-3:] == 'ies' and pword[:-3] + 'y' in words_lower_set:
    #         continue
    #
    #     elif pword[-3:] == 'ing' and pword[:-3] in words_lower_set:
    #         continue
    #
    #     elif ' ' in pword and pword.replace(' ', '') in words_set:
    #         continue
    #
    #     elif ' ' in pword and pword.replace(' ', '') in words_lower_set:
    #         continue
    #
    #     labels2.add(pword)
    # labels_new = labels2

    metadata_new['tags'] = list(labels_new)
    pixabay_labels_newA[key] = metadata_new

# *** Imagenet files ***

# DAG = IO.read_wordnet_heirarchy_file()
# print nx.ancestors(DG, 'n02691156')

synset_words_dict = IO.read_synset_words_file()

print('\n{:>13s}:{:>10s}{:>10s}{:>13s}'.format('', 'list', 'set', 'intersection'))
swords = [s for ss in synset_words_dict.values() for s in ss]
swords_set = set(swords)
words_set = swords_set.intersection(pwords_set)
print('{:13s}:{:10d}{:10d}{:13d}'.format('default', len(swords), len(swords_set), len(words_set)))
# lowercase
swords_lower = [s.lower() for s in swords]
swords_lower_set = set(swords_lower)
words_lower_set = swords_lower_set.intersection(pwords_set)
print('{:13s}:{:10d}{:10d}{:13d}\n\n'.format('lower', len(swords_lower), len(swords_lower_set), len(words_lower_set)))

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
for key, metadata in pixabay_labels.items():
    metadata_new = {}
    labels_new = metadata['tags']
    # if 'pelikan' in labels_new:
    #     if 'pelican' in labels_new:
    #         ii += 1
    #     else:
    #         print labels_new

    # if len(labels_new) > 0:
    #     labels_new = labels_new.difference(pixabay_blacklist)

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

    # for _, alias_list in pixabay_aliases.iteritems():
    #     if len(labels_new) > 0:
    #         labels_new = labels_new.difference(alias_list)

    labels2 = set()
    for pword in labels_new:
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

        if pword in pixabay_tallies and pixabay_tallies[pword] <= n_hits:
            continue

        labels2.add(pword)
    labels_new = labels2

    metadata_new['tags'] = list(labels_new)
    pixabay_labels_new[key] = metadata_new

# print 'found:', ii
max_tag_set = 0
for value in pixabay_labels.values():
    max_tag_set = max(max_tag_set, len(value['tags']))
max_tag_set += 1

tag_counts_old = {i: 0 for i in range(max_tag_set)}
for value in pixabay_labels.values():
    tag_counts_old[len(value['tags'])] += 1

tag_counts_newA = {i: 0 for i in range(max_tag_set)}
for key, value in pixabay_labels_newA.items():
    if len(value['tags']) > 8:
        print(key, value)
    tag_counts_newA[len(value['tags'])] += 1

tag_counts_new = {i: 0 for i in range(max_tag_set)}
for key, value in pixabay_labels_new.items():
    if len(value['tags']) > 6:
        print(key, value)
    tag_counts_new[len(value['tags'])] += 1
    # if len(value['tags']) > 0:
    #     set_a.update(value['tags'])

# print len(set_a)
total_tag_counts_old = sum(tag_counts_old.values())
total_tag_counts_newA = sum(tag_counts_newA.values())
total_tag_counts_new = sum(tag_counts_new.values())
for i in range(max_tag_set):
    if tag_counts_old[i] == 0 and tag_counts_newA[i] == 0 and tag_counts_new[i] == 0:
        continue
    tag_per_old = 100.0 * tag_counts_old[i] / total_tag_counts_old
    tag_per_newA = 100.0 * tag_counts_newA[i] / total_tag_counts_newA
    tag_per_new = 100.0 * tag_counts_new[i] / total_tag_counts_new
    print("{:3d} {:6d} {:6d} {:6d} {:10.3f} {:10.3f} {:10.3f}".format(i, tag_counts_old[i], tag_counts_newA[i], tag_counts_new[i], tag_per_old, tag_per_newA, tag_per_new))
print("All {:6d} {:6d} {:6d}".format(total_tag_counts_old, total_tag_counts_newA, total_tag_counts_new))


# dPWL = enchant.DictWithPWL('en_US', 'synonyms.yaml')
# replacer = CustomSpellingReplacer(dPWL)

syn_pe = wn.synset('physical_entity.n.01')


def get_pe_tag(word, posttag='*'):
    tag = ''
    for syn in wn.synsets(word, pos=wn.NOUN):
        for path in syn.hypernym_paths():
            if syn_pe in path:
                return posttag
    return tag


def get_pos_key(word, posttag=''):
    key = '0'
    if len(wn.synsets(word)) > 0:
        if len(wn.synsets(word, pos=wn.NOUN)) > 0:
            key = 'noun'+get_pe_tag(word, posttag='*')
        elif len(wn.synsets(word, pos=wn.VERB)) > 0:
            key = 'verb'
        elif len(wn.synsets(word, pos=wn.ADJ)) > 0:
            key = 'adj'
        elif len(wn.synsets(word, pos=wn.ADV)) > 0:
            key = 'adv'
        key += posttag
    return key

set_a = set()
dict_a = {}
blacklist_new = {}
key_list = ['0', 'dogs', 'cats', 'cars', 'nouns', 'verbs', 'adv', 'adj']
counts = {key: 0 for key in key_list}
counts_tally = {key: 0 for key in key_list}

for pword, ptally in pixabay_tallies.items():
    key = '0'

    if ptally <= n_hits:
        key = 'ignored'

        # handle cases

        # 'sail boat', 'sailboat'
        # 'sailing boat', 'sailboat'
        # 'sailing boats', 'sailboat'

    if key == '0':
        cword = pword.replace(' ', '_') if ' ' in pword else pword
        cword = yamlreplacer.replace(cword)
        key = get_pos_key(cword)

    if key == '0' and '_' in cword:
        wds = cword.split('_')
        join_key = str(len(wds))
        # key = 'multi'+str(len(wds))
        for joiner in ['-', '']:
            rword = cword.replace('_', joiner)
            key = get_pos_key(rword)
            # key = get_pos_key(cword, joiner+join_key)
            if key != '0':
                break

    if key == '0' and '-' in cword:
        wds = cword.split('-')
        join_key = str(len(wds))
        # key = 'multi'+str(len(wds))
        for joiner in ['_', '']:
            rword = cword.replace('-', joiner)
            key = get_pos_key(rword)
            # key = get_pos_key(cword, joiner+join_key)
            if key != '0':
                break

    # if key == '0' and '_' in cword:
    #     wds = cword.split('_')
    #     join_key = str(len(wds))
    #     # key = 'multi'+str(len(wds))
    #     for joiner in ['-', '']:
    #         rword = cword.replace('_', joiner)
    #         sword = stemmerporter.stem(rword)
    #         key = get_pos_key(sword)
    #         if key != '0':
    #             print '0', key, rword, sword, ptally
    #             break
    #         sword = stemmerlan.stem(rword)
    #         key = get_pos_key(sword)
    #         if key != '0':
    #             print '1', key, rword, sword, ptally
    #             break
    #
    # if key == '0' and '-' in cword:
    #     wds = cword.split('-')
    #     join_key = str(len(wds))
    #     # key = 'multi'+str(len(wds))
    #     for joiner in ['_', '']:
    #         rword = cword.replace('-', joiner)
    #         sword = stemmerporter.stem(rword)
    #         key = get_pos_key(sword)
    #         if key != '0':
    #             print '2', key, rword, sword, ptally
    #             break
    #         sword = stemmerlan.stem(rword)
    #         key = get_pos_key(sword)
    #         if key != '0':
    #             print '3', key, rword, sword, ptally
    #             break

    if key not in key_list:
        key_list.append(key)
        counts[key] = 0
        counts_tally[key] = 0

    # if key == 'noun':
    #     dict_a[pword] = cword
    #     set_a.update([pword])

    # if key in ('adj+', 'adv+'):
    #     print key, pword

    counts[key] += 1
    counts_tally[key] += ptally

    if key == '0':
        set_a.add(pword)
        blacklist_new[pword] = ptally


with open('data/pixabay_blacklist01.txt', 'w') as ofs:
    for bl, c in sorted(blacklist_new.items(), key=itemgetter(1), reverse=True):
        ofs.write("{}\t{}\n".format(c, bl))

c_sum = sum(counts.values())
c_tally_sum = sum(counts_tally.values())

fmt = '{:13s}:{:10d}{:10d}{:10.3f}{:10.3f}'
for k in key_list:
    if counts[k] == 0 and counts_tally[k] == 0:
        continue
    print(fmt.format(k, counts[k], counts_tally[k], 100.0 * counts[k] / c_total, 100.0 * counts_tally[k] / c_tally_total))
print(fmt.format('sum', c_sum, c_tally_sum, 100.0 * c_sum / c_total, 100.0 * c_tally_sum / c_tally_total))


print('\n')


set_b = set()
blacklist_new = {}
key_list = ['0', 'dogs', 'cats', 'cars', 'noun', 'verb', 'adv', 'adj', 'blacklist', 'alias', 'ws']
counts = {key: 0 for key in key_list}
counts_tally = {key: 0 for key in key_list}

print_delims = {' ': '_', '-': '-'}
delims = [(' ', '', '_1'), (' ', '-', '_2'), ('-', ' ', '_3'), ('-', '', '_4')]
suffi = [('s', ''), ('es', ''), ('ies', 'y'), ('ing', ''), ('ed', '')]

for pword, ptally in pixabay_tallies.items():
    key = '0'

    # if pword == "coca-cola":
    #     print pword

    # if pword in dog_breeds:
    #     key = 'dogs'
    #
    # elif pword in cat_breeds:
    #     key = 'cats'
    #
    # elif pword in car_makes:
    #     key = 'cars'

    if pword in wordnet_nouns:
        key = 'noun'

    elif pword in wordnet_verbs:
        key = 'verb'

    elif pword in wordnet_adv:
        key = 'adv'

    elif pword in wordnet_adj:
        key = 'adj'

    # elif pword in pixabay_blacklist:
    #     key = 'blacklist'

    elif pword in words_set:
        key = 'ws'

    elif pword in words_lower_set:
        key = 'wsl'

    # else:
    #     for _, alias_list in pixabay_aliases.iteritems():
    #         if pword in alias_list:
    #             key = 'alias'
    #             break

    # if iwc == 1:
    #     key = 'alias'
    #     continue

    # if key == 'noun':
    #     set_b.add(pword)
    #     if pword not in set_a:
    #         print pword, ptally

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
        # if len(ws_cv) > 1:
        #     print 'ws', pword, ws_cv, ptally
        for dkey, word in candidates.items():
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
        # if len(wsl_cv) > 1:
        #     print 'wsl', pword, wsl_cv, ptally
        for dkey, word in candidates.items():
            if word in words_lower_set:
                key = 'wsl' + dkey

    if key not in key_list:
        key_list.append(key)
        counts[key] = 0
        counts_tally[key] = 0

    counts[key] += 1
    counts_tally[key] += ptally

    if key == '0':
        set_b.add(pword)
        blacklist_new[pword] = ptally


with open('data/pixabay_blacklist00.txt', 'w') as ofs:
    for bl, c in sorted(blacklist_new.items(), key=itemgetter(1), reverse=True):
        ofs.write("{}\t{}\n".format(c, bl))

# print set_a.symmetric_difference(set_b)

for pword in list(set_a.difference(set_b)):
    print(pword, pixabay_tallies[pword])

c_sum = sum(counts.values())
c_tally_sum = sum(counts_tally.values())

fmt = '{:13s}:{:10d}{:10d}{:10.3f}{:10.3f}'
for k in key_list:
    if counts[k] == 0 and counts_tally[k] == 0:
        continue
    print(fmt.format(k, counts[k], counts_tally[k], 100.0 * counts[k] / c_total, 100.0 * counts_tally[k] / c_tally_total))
print(fmt.format('sum', c_sum, c_tally_sum, 100.0 * c_sum / c_total, 100.0 * c_tally_sum / c_tally_total))

print('\n')

