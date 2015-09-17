from __future__ import print_function
import os
import glob
import re
import codecs
from operator import itemgetter
import shutil

def clean_text(text):
    text = text.lower()
    text = ' '.join(text.split())
    text = ''.join([c for c in text if c.isalpha() or c.isspace() or c.isdigit()])
    return text

# first target data (uniformize filenames):
dirty_path = '../data/soldier_letters_dirty'
clean_path = '../data/soldier_letters'

if os.path.isdir(clean_path):
    shutil.rmtree(clean_path)
os.mkdir(clean_path)

for filename in glob.glob(dirty_path+'/*.txt'):
    metadata = os.path.splitext(os.path.basename(filename))[0].split('_')
    text_id, date, village, scribe, _, _, _ = metadata
    scribe = ''.join([n.lower().capitalize().replace('.', '') for n in scribe.split('_')])
    new_filename = clean_path+'/'+('_'.join([scribe, text_id, village, date]))+'.txt'
    with codecs.open(new_filename, 'w', 'utf8') as new_f:
        with codecs.open(filename) as old_f:
            text = old_f.read()
            text = clean_text(text)
            new_f.write(text)

# sailing data:
dirty_path = '../data/soldier_sailing_dirty'
clean_path = '../data/soldier_sailing'

if os.path.isdir(clean_path):
    shutil.rmtree(clean_path)
os.mkdir(clean_path)

nb_unknown = 1
for filename in glob.glob(dirty_path+'/*.txt'):
    metadata = os.path.splitext(os.path.basename(filename))[0].split('-')
    name, text_id = metadata[3], metadata[0]
    name = ''.join([n.lower().capitalize().replace('.', '') for n in name.split()])
    if name == "ONBEKEND":
        name += str(nb_unknown)
        nb_unknown += 1
    new_filename = clean_path+'/'+name+'_'+text_id+'.txt'
    print(new_filename)
    with codecs.open(new_filename, 'w', 'utf8') as new_f:
        with codecs.open(filename) as old_f:
            text = old_f.read()
            text = clean_text(text)
            print(text[:1000])
            new_f.write(text)

# bab data:
dirty_path = '../data/soldier_bab_dirty'
clean_path = '../data/soldier_bab'
if os.path.isdir(clean_path):
    shutil.rmtree(clean_path)
os.mkdir(clean_path)

for idx, filename in enumerate(sorted(glob.glob(dirty_path+'/*.txt'))):
    new_filename = clean_path+'/'+str(idx)+'_'+str(idx)+'.txt'
    print(new_filename)
    with codecs.open(new_filename, 'w', 'utf8') as new_f:
        with codecs.open(filename) as old_f:
            text = old_f.read()
            text = clean_text(text)
            print(text[:1000])
            new_f.write(text)


# dirty data:
dirty_path = '../data/soldier_armen_dirty'
clean_path = '../data/soldier_armen'
if os.path.isdir(clean_path):
    shutil.rmtree(clean_path)
os.mkdir(clean_path)

for idx, filename in enumerate(sorted(glob.glob(dirty_path+'/*.txt'))):
    new_filename = clean_path+'/'+str(idx)+'_'+str(idx)+'.txt'
    print(new_filename)
    with codecs.open(new_filename, 'w', 'utf8') as new_f:
        with codecs.open(filename) as old_f:
            text = old_f.read()
            text = clean_text(text)
            print(text[:1000])
            new_f.write(text)

