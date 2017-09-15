#!/usr/bin/env python3
#-*- coding: UTF-8 -*-
#Created by r05922145@ntu.edu.tw at 2017/09/15

import sys

def word_count(path):
    word_list_count = []
    count_dic = {}
    with open(path, 'r') as f:
        word_list = f.read().split(' ')
        for word in word_list:
            word = word.strip('\n')
            if word not in word_list_count:                 
                word_list_count.append(word)
                count_dic[word] =  1
            else:
                count_dic[word] = count_dic[word] + 1
    #print (count_dic, '\n') 
    #print (word_list_count, '\n')   
    return word_list_count, count_dic

def write_file(word_list_count, count_dic):
        with open('Q1.txt', 'w') as f:
            for word in word_list_count:
                f.write(word)
                f.write(' ')
                f.write(str(word_list_count.index(word)))
                f.write(' ')
                f.write(str(count_dic[word]))
                if word_list_count.index(word) != (len(word_list_count) - 1):
                    f.write('\n')

if __name__ == '__main__':
    path = sys.argv[1]
    word_list_count, count_dic = word_count(path)
    write_file(word_list_count, count_dic)
