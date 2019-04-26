import sys
import numpy as np

'''
generates three different files: 

1) a .sites file (formatted for ARGweaver and CLUES)
2) a .trees file for each segsites
3) an .h5 file that holds the trajectory
'''

def convert(filename,length,numIndividuals,outfilename):
	
        SUPPRESS_TREES = False #False 
        SUPPRESS_TRAJ = False #True
	
        gmax = 2*10**5
        n = 0
        num_iter = 0
        f = open(filename,'r')
        for line in f: 
                if line[:4] == 'Freq':
                        if SUPPRESS_TRAJ:
                                continue
                        data = []
                        trajLine = f.readline()
                        trajLine = f.readline()
                        while trajLine.rstrip() != '':
                                cols = trajLine.rstrip().split('\t')
                                freq = float(cols[1])
                                data.append(freq)
                                trajLine = f.readline()
                        #data = data[::-1]
                        data += [0 for _ in range(gmax - len(data))]      		
                        np.savetxt(outfilename+'.i_'+str(num_iter)+'.traj',np.array(data))
                elif line[0] == '[':
                        prev_p = -999
                        treeDict = {} 
                        treeLine = line
                        posn = 0
                        if SUPPRESS_TREES:
                                continue
                        while treeLine.rstrip() != '':
                                prev_posn = posn
                                posn = length/nsites*int(treeLine.split('[')[1].split(']')[0]) + prev_posn
                                treeDict[(prev_posn,posn)] = treeLine.rstrip().split(']')[1]
                                treeLine = f.readline()
                        #outTrees = open(outfilename+'.i_'+str(num_iter)+'.trees','w')
                        #tree = line.split(']')[1].rstrip()
                        #outTrees.write(tree) 
                elif line[:8] == 'segsites':
                        out = open(outfilename+'.i_'+str(num_iter)+'.sites','w')
                        out.write('NAMES\t'+'\t'.join([str(i) for i in range(numIndividuals)])+'\n')
                        out.write('REGION\tchr\t1\t'+str(length)+'\n')

                        s = int(line.rstrip().split(' ')[1])
                        n = numIndividuals
                        genotypes = np.zeros((n,s))
                        continue
                if n != 0:
                        if line[0] == 'p':
                                positions = [int(float(length) * float(p)) for p in line.rstrip().split(' ')[1:]]
                                #print(positions)
                                if SUPPRESS_TREES:
                                        continue
                                
                                for p in [100000]:
                                	print(p)        
                                	for key in treeDict.keys():
                                		a = key[0]
                                		b = key[1]
                                		print(a,b)
                                		if a <= p and p < b:
                                			outTrees = open(outfilename+'.i_'+str(num_iter)+'.posn_'+str(p)+'.trees','w') 
                                			outTrees.write(treeDict[key])
                                			continue
                        else:
                                try:
                                        genotypes[numIndividuals-n,:] = [0 if x == '0' else 1 for x in line.rstrip()]
                                        #print(genotypes[numIndividuals-n,:])
                                except:
                                        #print(s,len(line),line)
                                        raise ValueError
                                n -= 1
                                if n == 0:
                                        #print(genotypes[:,0])
                                        for (j,p) in enumerate(positions):
                                                if np.sum(genotypes[:,j]) > 0 and np.sum(genotypes[:,j]) < numIndividuals and prev_p != p:
                                                        #print('hi')
                                                        out.write(str(p)+'\t'+''.join(['G' if x == 1 else 'A' for x in genotypes[:,j]])+'\n')
                                                prev_p = p
                                        num_iter += 1
                                        out.close()
        return


filename = sys.argv[1]
length = int(sys.argv[2])
nsites = int(sys.argv[3]) 
numIndividuals = int(sys.argv[4])
outfilename = sys.argv[5]

convert(filename,length,numIndividuals,outfilename)
