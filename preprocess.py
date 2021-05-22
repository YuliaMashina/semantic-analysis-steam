from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import re
import nltk
import pkg_resources
from symspellpy import SymSpell, Verbosity
from nltk.tokenize import word_tokenize
from language_detector import detect_language

sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
if sym_spell.word_count:
    pass
else:
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

def f_stopw(w_list):
    return [word for word in w_list if word not in en_stop]

def regex_preprocess(s):
    s = re.sub(r'\(.*?\)', '. ', s)
    s = re.sub(r'\W+?\.', '.', s)
    s = re.sub(r'(\.|\?|!)(\w)', r'\1 \2', s)
    s = re.sub(r' ing ', ' ', s)
    s = re.sub(r'product received for free[.| ]', ' ', s)
    s = re.sub(r'(.{2,}?)\1{1,}', r'\1', s)
    s = re.sub(r'([a-z])([A-Z])', r'\1\. \2', s)
    s = s.lower()
    s = re.sub(r'&gt|&lt', ' ', s)
    s = re.sub(r'([a-z])\1{2,}', r'\1', s)
    s = re.sub(r'([\W+])\1{1,}', r'\1', s)
    s = re.sub(r'\*|\W\*|\*\W', '. ', s)
    return s.strip()

def preprocess_word(s):
    if not s:
        return None
    w_list = word_tokenize(s)
    w_list = filter_punctuation(w_list)
    w_list = filter_nouns(w_list)
    w_list = correct_typos(w_list)
    w_list = f_stem(w_list)
    w_list = f_stopw(w_list)

    return w_list

def f_stem(w_list):
    return [p_stemmer.stem(word) for word in w_list]


en_stop = get_stop_words('en')
en_stop.append('game')
en_stop.append('time')
en_stop.append('play')
en_stop.append('player')


def f_lan_detect(s):
    return detect_language(s) in {'English'}

def filter_nouns(w_list):
    return [word for (word, pos) in nltk.pos_tag(w_list) if pos[:2] == 'NN']

def filter_punctuation(w_list):
    return [word for word in w_list if word.isalpha()]

def correct_typos(w_list):
    w_list_fixed = []
    for word in w_list:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=3)
        if suggestions:
            w_list_fixed.append(suggestions[0].term)
    return w_list_fixed

p_stemmer = PorterStemmer()

def preprocess_sentence(rw):
    s = regex_preprocess(rw)
    if not f_lan_detect(s):
        return None
    return s