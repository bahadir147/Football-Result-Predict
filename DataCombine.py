# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 22:08:44 2019

@author: Bahadir
"""

fout=open("out.csv","a")
# first file:
for line in open("T1 (0).csv"):
    fout.write(line)
# now the rest:    
for num in range(1,12):
    f = open("T1 ("+str(num)+").csv")
    f.__next__() # skip the header
    for line in f:
         fout.write(line)
    f.close() # not really needed
fout.close()