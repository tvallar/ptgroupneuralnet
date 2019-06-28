from root_numpy import root2array, tree2array
from root_numpy import testdata

filename = testdata.get_filepath('test.root')

# Convert a TTree in a ROOT file into a NumPy structured array
arr = root2array(filename, 'tree')
# The TTree name is always optional if there is only one TTree in the file

# Or first get the TTree from the ROOT file
import ROOT
rfile = ROOT.TFile(filename)
intree = rfile.Get('tree')

# and convert the TTree into an array
array = tree2array(intree)
