import os
import math
import sys
import argparse

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--lang', help='which language to crawl', required=True)
    args = vars(parser.parse_args())
    lang = args['lang']

    #run generate 
    print "generating Images..."
    os.system('python generateCharImg.py -l ' + lang + ' -p ../data/')

    #build Img t7
    print "building img t7..."
    path = './' + lang + '/img'
    num_chars = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    os.system('th buildImgSet.lua -l ' + lang + ' -n ' + str(num_chars))
    
    
    #build dataset t7
    splits = ['train', 'val', 'test']
    num_classes = 12
    rho = 10
    for split in splits:
        print "building " + split + " t7..."
        os.system('python append_zeros.py -l ' + lang + ' -s ' + split)
        fileName = './' + lang + '/' + lang + '_' + split + '.txt'
        num_instances = len(open(fileName, 'r').readlines())
        os.system('th buildDataset.lua -l ' + lang + ' -c ' + str(num_classes) + ' -n ' + str(num_instances) + ' -r ' + str(rho) + ' -s ' + split) 


    
