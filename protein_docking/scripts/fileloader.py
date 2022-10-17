################################################################################
# fileloader.py
################################################################################
'''
This file (fileloader.py) has functions to load fasta files and pdb files for
this project. It is used by other python scripts in this subdirectory 'scripts'.

You can import it into other python scripts by including this in your python 
script:

import fileloader
...

-CMV 4.19.19
'''
################################################################################
# LIBRARIES/IMPORTS
################################################################################
import numpy as np

################################################################################
# PATHS
################################################################################
fasta_t0860 = '../fasta/5fjl.fasta'
fasta_t0893 = '../fasta/5idj.fasta'
pdb_t0860 = '../pdb/true/5fjl.pdb'
pdb_t0893 = '../pdb/true/5idj.pdb'

################################################################################
# FUNCTIONS
################################################################################
'''
This function loads a fasta file with a sequence int from the given path and 
returns a single string with the sequence. The usual stuff...
'''
def load_fasta(file):
	fp = open(file,'r')
	lines = list(fp)
	name = lines[0][1:5]
	fp.close()
	s = ''
	for line in lines: 
		if line.rstrip()[0] != '>' and line.rstrip()[-1]!='':
			s += line.rstrip()
	return name, s

'''
This function loads a pdb file from the given path, and loads the 
x,y,z coordindates into a numpy array with three columns x,y,z and L rows
with L being the sequence length.
'''
def load_pdb(file):
	fp = open(file,'r')
	coordinates  = []
	rotations    = []
	translations = []
	count = 0
	for line in fp:
		if line[0:4]=='ATOM':
			coordinates.append([float(line[30:38]),float(line[38:46]),float(line[46:54])])
		if line[0:10]=='REMARK 350' and line[13:18]=='BIOMT':
			if count==2:
				rotations.append(rotation)
				translations.append(translation)
				count = 0
			if count == 0:
				rotation = np.zeros((3,3))
				translation = np.zeros(3)
				rotation[count,:]  = [line[23:33],line[33:43],line[43:54]]
				translation[count] = line[59:68]
			else:
	fp.close()
	coordinates = np.array(coordinates)
	rotations = np.array(rotations)
	translations = np.array(translations)
	return coordinates,rotations,translations

def main():
	coords, r, t = load_pdb(pdb_t0860)
	print(r.shape)


if __name__=='__main__':
	main()