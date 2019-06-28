#!/bin/sh
./bosread /run/media/ptgroup/MyBookDuo/josh_script/sigmazero/*.a1c /run/media/ptgroup/MyBookDuo/josh_script/sigmazero_1/*.a1c
mv outPutFile.root sigmazero_bosread.root 
./bosread /run/media/ptgroup/MyBookDuo/josh_script/kstarpizero/*.a1c /run/media/ptgroup/MyBookDuo/josh_script/kstarpizero_1/*.a1c
mv outPutFile.root kstarpizero_bosread.root
~/pacordova/code/newchannel/merge_new/merge_sig sigmazero_bosread.root
mv mergeplot.root sigmazero_mergeplot.root
~/pacordova/code/newchannel/merge_new/merge_bg kstarpizero_bosread.root
mv mergeplot.root kstarpizero_mergeplot.root
./prep_trees sigmazero_mergeplot.root kstarpizero_mergeplot.root 30000
#~/pacordova/code/testground/merge_old/MergePlot sigmazero_bosread.root
#~/pacordova/code/testground/merge_old/MergePlot kstarpizero_bosread.root
