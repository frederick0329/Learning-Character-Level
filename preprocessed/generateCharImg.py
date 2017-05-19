#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import io
import codecs
import os
import pickle
import json
import argparse
from draw import draw
from bs4 import BeautifulSoup


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--lang', help='which language to crawl', required=True)
    parser.add_argument('-p','--prefix', help='which language to crawl', required=True)
    args = vars(parser.parse_args())
    utf2idx = {}
    idx2utf = []
    cat2idx = {}
    idx2cat = []
    cat_idx = 1
    char_idx = 1
    prefix = args['prefix']
    lang = args['lang']
    paths = ["train.txt", "val.txt", "test.txt"]
   
    langs = ['zh_traditional', 'zh_simplified', 'ja', 'ko']

    if not os.path.exists(lang):
        os.makedirs(lang)

    if not os.path.exists(lang + '/img'):
        os.makedirs(lang + '/img')
 
    for path in paths:
        with io.open(prefix + lang + '_' + path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                category, content = line.split("\t")

                if category not in cat2idx:
                    cat2idx[category] = cat_idx
                    idx2cat.append(category)
                    cat_idx += 1
                for c in list(content.rstrip('\n')):
                    if c not in utf2idx:
                        utf2idx[c] = char_idx
                        idx2utf.append(c)
                        draw(c , char_idx, lang + "/img/", lang)
                        char_idx += 1
    


    pickle.dump(cat2idx, open( lang + "/cat2idx.pkl", "wb" ))
    pickle.dump(idx2cat, open( lang + "/idx2cat.pkl", "wb" ))
    pickle.dump(utf2idx, open( lang + "/utf2idx.pkl", "wb" ))
    pickle.dump(idx2utf, open( lang + "/idx2utf.pkl", "wb" ))



