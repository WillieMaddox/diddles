# -*- coding: utf-8 -*-
import nltk
import enchant
import unicodedata
from bs4 import UnicodeDammit
from nltk.stem import PorterStemmer
from nltk.stem import RegexpStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import names
from nltk.corpus import words
from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic
from nltk.corpus import genesis
from nltk.corpus import stopwords
from nltk.metrics import *
from nltk.metrics import edit_distance
from nltk.tokenize import word_tokenize
from replacers import RepeatReplacer
from replacers import RegexpReplacer
from replacers import SpellingReplacer
from replacers import CustomSpellingReplacer
from replacers import AntonymReplacer
from replacers import AntonymWordReplacer

# 1_33
training = 'PERSON OTHER PERSON OTHER OTHER ORGANIZATION'.split()
testing = 'PERSON OTHER OTHER OTHER OTHER OTHER'.split()
trainset = set(training)
testset = set(testing)
print(accuracy(training, testing))
print(precision(trainset, testset))
print(f_measure(trainset, testset))
print(recall(trainset, testset))

# 4_3
print(nltk.help.upenn_tagset('VB.*'))

# 4_14
print(len(names.words('male.txt')))
print(len(names.words('female.txt')))

# 4_15
print(words.fileids())
print(len(words.words('en')))
print(len(words.words('en-basic')))


# 8_1
def not_stopwords(text):
    en_stopwords = stopwords.words('english')
    content = [w for w in text if w.lower() not in en_stopwords]
    return len(content) / len(text)
print(stopwords.words('english'))
print(not_stopwords(nltk.corpus.reuters.words()))
print(not_stopwords(nltk.corpus.inaugural.words()))

print wordnet.synsets('cat')
print wordnet.synsets('cat', pos=wordnet.VERB)
cat = wordnet.synset('cat.n.01')
dog = wordnet.synset('dog.n.01')

print(cat.definition())
print(len(cat.examples()))
print(cat.lemmas())
print([str(lemma.name()) for lemma in cat.lemmas()])
print(wordnet.lemma('cat.n.01.cat').synset())

print(sorted(wordnet.langs()))
print(cat.lemma_names('ita'))
print(sorted(cat.lemmas('dan')))
print(sorted(cat.lemmas('por')))
print(len(wordnet.all_lemma_names(pos='n', lang='jpn')))
print(cat.hypernyms())
print(cat.hyponyms())
print(cat.member_holonyms())
print(cat.root_hypernyms())
print(cat.lowest_common_hypernyms(dog))

# NLP_w_Python Ch 1

print '========================================='
print 'Looking up a Synset for a Word in WordNet'
print '========================================='

syn = wordnet.synsets('cookbook')[0]
print syn.name() == 'cookbook.n.01'
print syn.definition() == 'a book of recipes and cooking directions'

print wordnet.synset('cookbook.n.01')  # == "Synset('cookbook.n.01')"

print wordnet.synsets('cooking')[0].examples()  # == "['cooking can be a great art', 'people are needed who have experience in cookery', 'he left the preparation of meals to his wife']"

print syn.hypernyms()  # == "[Synset('reference_book.n.01')]"

print syn.hypernyms()[0].hyponyms()  # == "[Synset('encyclopedia.n.01'), Synset('directory.n.01'), Synset('source_book.n.01'), Synset('handbook.n.01'), Synset('instruction_book.n.01'), Synset('cookbook.n.01'), Synset('annual.n.02'), Synset('atlas.n.02'), Synset('wordbook.n.01')]"

print syn.root_hypernyms()  # == "[Synset('entity.n.01')]"

print syn.hypernym_paths()  # == "[[Synset('entity.n.01'), Synset('physical_entity.n.01'), Synset('object.n.01'), Synset('whole.n.02'), Synset('artifact.n.01'), Synset('creation.n.02'), Synset('product.n.02'), Synset('work.n.02'), Synset('publication.n.01'), Synset('book.n.01'), Synset('reference_book.n.01'), Synset('cookbook.n.01')]]"

print syn.pos() == 'n'

print len(wordnet.synsets('great')) == 7
print len(wordnet.synsets('great', pos='n')) == 1
print len(wordnet.synsets('great', pos='a')) == 6

print '========================================='
print 'Looking up Lemmas and Synonyms in WordNet'
print '========================================='

lemmas = syn.lemmas()
print len(lemmas) == 2
print lemmas[0].name() == 'cookbook'
print lemmas[1].name() == 'cookery_book'
print lemmas[0].synset() == lemmas[1].synset()

print [lemma.name() for lemma in syn.lemmas()] == ['cookbook', 'cookery_book']

synonyms = []
for syn in wordnet.synsets('book'):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name)

print len(synonyms) == 38
print len(set(synonyms)) == 25

gn2 = wordnet.synset('good.n.02')
print gn2.definition() == 'moral excellence or admirableness'
evil = gn2.lemmas()[0].antonyms()[0]
print evil.name() == 'evil'
print evil.synset().definition() == 'the quality of being morally wrong in principle or practice'

ga1 = wordnet.synset('good.a.01')
print ga1.definition() == 'having desirable or positive qualities especially those suitable for a thing specified'
bad = ga1.lemmas()[0].antonyms()[0]
print bad.name() == 'bad'
print bad.synset().definition() == 'having undesirable or negative qualities'

print '====================================='
print 'Calculating WordNet Synset Similarity'
print '====================================='

lion = wordnet.synset('lion.n.01')

print(lion.path_similarity(cat))
# print(lion.lch_similarity(cat))
print(lion.wup_similarity(cat))

brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')
genesis_ic = wordnet.ic(genesis, False, 0.0)
print(lion.res_similarity(cat, brown_ic))
print(lion.res_similarity(cat, genesis_ic))
print(lion.jcn_similarity(cat, brown_ic))
print(lion.jcn_similarity(cat, genesis_ic))
print(lion.lin_similarity(cat, semcor_ic))

cb = wordnet.synset('cookbook.n.01')
ib = wordnet.synset('instruction_book.n.01')
print cb.wup_similarity(ib) == 0.91666666666666663

ref = cb.hypernyms()[0]
print cb.shortest_path_distance(ref) == 1
print ib.shortest_path_distance(ref) == 1
print cb.shortest_path_distance(ib) == 2

dog = wordnet.synsets('dog')[0]
print dog.wup_similarity(cb) == 0.38095238095238093

print dog.common_hypernyms(cb)  # == "[Synset('object.n.01'), Synset('whole.n.02'), Synset('physical_entity.n.01'), Synset('entity.n.01')]"

cook = wordnet.synset('cook.v.01')
bake = wordnet.synset('bake.v.02')
print cook.wup_similarity(bake)  # == 0.75

print cb.path_similarity(ib) == 0.33333333333333331
print cb.path_similarity(dog) == 0.071428571428571425
print cb.lch_similarity(ib) == 2.5389738710582761
print cb.lch_similarity(dog) == 0.99852883011112725

print '=============='
print 'Stemming Words'
print '=============='

# 3_1
stemmerporter = PorterStemmer()
print stemmerporter.stem('cooking') == 'cook'
print stemmerporter.stem('cookery') == 'cookeri'
print(stemmerporter.stem('working'))
print(stemmerporter.stem('happiness'))

# 3_2
stemmerlan = LancasterStemmer()
print stemmerlan.stem('cooking') == 'cook'
print stemmerlan.stem('cookery') == 'cookery'
print(stemmerlan.stem('working'))
print(stemmerlan.stem('happiness'))
print(stemmerlan.stem('achievement'))

# 3_3
stemmerregexp = RegexpStemmer('ing')
print stemmerregexp.stem('cooking') == 'cook'
print stemmerregexp.stem('cookery') == 'cookery'
print stemmerregexp.stem('ingleside') == 'leside'
print(stemmerregexp.stem('working'))
print(stemmerregexp.stem('happiness'))
print(stemmerregexp.stem('pairing'))

# 3_4
print(SnowballStemmer.languages) == ('danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian', 'norwegian', 'porter', 'portuguese', 'romanian', 'russian', 'spanish', 'swedish')
stemmerspanish = SnowballStemmer('spanish')
print stemmerspanish.stem('hola') == 'hol'
print(stemmerspanish.stem('comiendo'))
stemmerfrench = SnowballStemmer('french')
print(stemmerfrench.stem('manger'))

print '=============================='
print 'Lemmatising Words with WordNet'
print '=============================='

lemmatizer = WordNetLemmatizer()
print lemmatizer.lemmatize('cooking') == 'cooking'
print lemmatizer.lemmatize('cooking', pos='v') == 'cook'
print lemmatizer.lemmatize('cookbooks') == 'cookbook'

print(lemmatizer.lemmatize('working'))
print(lemmatizer.lemmatize('working', pos='v'))
print(lemmatizer.lemmatize('works'))

print(stemmerporter.stem('happiness'))
print(lemmatizer.lemmatize('happiness'))

print stemmerporter.stem('believes') == 'believ'
print lemmatizer.lemmatize('believes') == 'belief'

print stemmerporter.stem('buses') == 'buse'
print lemmatizer.lemmatize('buses') == 'bus'
print stemmerporter.stem('bus') == 'bu'

print '============================================'
print 'Replacing Words Matching Regular Expressions'
print '============================================'

replacer = RegexpReplacer()
print replacer.replace("can't is a contraction") == 'cannot is a contraction'
print replacer.replace("I should've done that thing I didn't do") == 'I should have done that thing I did not do'

print word_tokenize("can't is a contraction") == ['ca', "n't", 'is', 'a', 'contraction']
print word_tokenize(replacer.replace("can't is a contraction")) == ['can', 'not', 'is', 'a', 'contraction']

print '============================='
print 'Removing Repeating Characters'
print '============================='

replacer = RepeatReplacer()
print replacer.replace('looooove') == 'love'
print replacer.replace('oooooh') == 'ooh'
print replacer.replace('goose') == 'goose'

# 1_34
print(edit_distance("relate", "relation"))
print(edit_distance("suggestion", "calculation"))

# 1_35
X = {10, 20, 30, 40}
Y = {20, 30, 60}
print(jaccard_distance(X, Y))

print '================================'
print 'Spelling Correction with Enchant'
print '================================'

replacer = SpellingReplacer()
print replacer.replace('cookbok') == 'cookbook'

d = enchant.Dict('en')
print d.suggest('languege')  # == ['language', 'languages', 'languor', "language's"]

print edit_distance('language', 'languege') == 1
print edit_distance('language', 'languor') == 3

print enchant.list_languages()  # == ['en', 'en_CA', 'en_GB', 'en_US']

dUS = enchant.Dict('en_US')
dGB = enchant.Dict('en_GB')
us_replacer = SpellingReplacer('en_US')
gb_replacer = SpellingReplacer('en_GB')

print dUS.check('theater')
print not dGB.check('theater')
print us_replacer.replace('theater') == 'theater'
print gb_replacer.replace('theater') == 'theatre'

dPWL = enchant.DictWithPWL('en_US', 'mywords.txt')
replacer = CustomSpellingReplacer(dPWL)

print not dUS.check('nltk')
print dPWL.check('nltk')
print replacer.replace('nltk') == 'nltk'

print '================================='
print 'Replacing Negations with Antonyms'
print '================================='

replacer = AntonymReplacer()
print replacer.replace('good') is None
print replacer.replace('uglify') == 'beautify'
sent = ["let's", 'not', 'uglify', 'our', 'code']
print replacer.replace_negations(sent) == ["let's", 'beautify', 'our', 'code']

replacer = AntonymWordReplacer({'evil': 'good'})
print replacer.replace_negations(['good', 'is', 'not', 'evil']) == ['good', 'is', 'good']

# NLP_w_Python Ch 9

print '============================================'
print 'Detecting and Converting Character Encodings'
print '============================================'

print unicodedata.normalize('NFKD', u'abcd\xe9').encode('ascii', 'ignore') == b'abcde'
print UnicodeDammit(u'abcd\xe9').unicode_markup == u'abcdé'



