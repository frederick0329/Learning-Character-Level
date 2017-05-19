#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import io
import codecs
import pickle
import argparse
from draw import draw
from bs4 import BeautifulSoup
from random import shuffle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--lang', help='lang', required=True)
    parser.add_argument('-s','--split', help='split', required=True)
    args = vars(parser.parse_args())
    lang = args['lang']
    split = args['split']
    #lang = "zh_traditional"
    #split = "val"
    utf2idx = pickle.load(open(lang + "/utf2idx.pkl", "r"))
    idx2utf = pickle.load(open(lang + "/idx2utf.pkl", "r" ))
    cat2idx = pickle.load(open(lang + "/cat2idx.pkl", "r"))
    idx2cat = pickle.load(open(lang + "/idx2cat.pkl", "r" ))
    path = "../data/" + lang + "_" + split + ".txt"
    output_path = "./" + lang + '/' + lang + "_" + split + ".txt"
    f_out = open(output_path, "w")
    count = 0
    total_len = 0
    numIns = 0
    with io.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            numIns += 1
            if lang == 'multilingual':
                category, content, lang_idx = line.split("\t")
            else:
                category, content = line.split("\t")
            f_out.write(str(cat2idx[category]))
            f_out.write("\t")
            chars = list(content.rstrip('\n'))
            total_len += len(chars)
            for i in xrange(len(chars)):
                if chars[i] not in utf2idx:
                    f_out.write(str(len(idx2utf)))
                    count += 1
                else:
                    f_out.write(str(utf2idx[chars[i]]))
                if i < (len(chars) - 1 ):
                    f_out.write(",")
            f_out.write("\t")
            for i in xrange(len(chars)):
                f_out.write("0")
            f_out.write("\n")

    f.close()
    print split, "statistics"
    print "number of instance:", numIns
    print "avg len of sentence:", float(total_len) / numIns
    print "number of category:", len(idx2cat)
    print "number of chars:", len(idx2utf) 
    print "-"*20


