#!/usr/bin/python
from multiprocessing import Process
from multiprocessing import Pool
import os,time,sys
from subprocess import Popen

def complete(ls):
    proc=Popen("gamp2part -r56855 -o./block_"+ls+".part -T -S-0.321,0.378,-0.254,0.407 -z-110,-70 ./block_"+ls+".gamp",shell=True,executable=os.environ.get('SHELL','/bin/bash'),env=os.environ)
    proc.wait()
    proc=Popen("gsim_bat -ffread ffread.g12 -kine 1 -mcin ./block_"+ls+".part -bosout ./block_"+ls+".gsim -trig 2000000",shell=True,executable=os.environ.get('SHELL','/bin/bash'),env=os.environ)
    proc.wait()
    proc=Popen("gpp -Y -s -S -a2.73 -b1.7 -c1.93 -f1 -R56855 -P0x7f -o./block_"+ls+".gpp ./block_"+ls+".gsim -A/u/home/clasg12/gpp_tagger_profile.bos",shell=True,executable=os.environ.get('SHELL','/bin/bash'),env=os.environ)
    proc.wait()
    proc=Popen("a1c -T4 -sa -ct1930 -cm0 -cp0 -X0 -d1 -F -P0x1bff -z0,0,-90 -Aprlink_tg-90pm30.bos -o./block_"+ls+".a1c ./block_"+ls+".gsim",shell=True,executable=os.environ.get('SHELL','/bin/bash'),env=os.environ)
    proc.wait()


if __name__ == '__main__':
    threads = 24
    p = Pool(threads)
    print "Start"
    map_list = []
    for i in range(threads):
        map_list.append(str(i))
        proc=Popen("genr8 -M50000 -B4.5,5.45 -L0.0 -U3.0 -o./block_"+str(i)+".gamp < ./kstarpizero.input",shell=True,executable=os.environ.get('SHELL','/bin/bash'),env=os.environ)
	proc.wait()
	time.sleep(1)
    p.map(complete, map_list)
