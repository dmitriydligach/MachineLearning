#!/usr/local/python/bin/python -B

import os, scipy.integrate, numpy, sys

DEFAULTDIR = 'curves'

def area(file):
  """Difference: selective curve - random curve"""

  randsizes = []
  randaccs = []

  selsizes = []
  selaccs = []

  for line in open(file):
    randsize, randacc, selsize, selacc = line.split()

    randsizes.append(float(randsize))
    randaccs.append(float(randacc))

    selsizes.append(float(selsize))
    selaccs.append(float(selacc))

  randarea = scipy.integrate.trapz(randaccs, randsizes)
  selarea = scipy.integrate.trapz(selaccs, selsizes)

  return selarea - randarea

def area2(file):
  """Difference: selective curve - random curve. Use relative sizes."""

  randsizes = []
  randaccs = []

  selsizes = []
  selaccs = []

  for line in open(file):
    randsize, randacc, selsize, selacc = line.split()
    
    randsizes.append(float(randsize))
    randaccs.append(float(randacc))

    selsizes.append(float(selsize))
    selaccs.append(float(selacc))

  if len(randsizes) == 0:
    return 0.0 # nothing in this file
  
  # use actual training set size (e.g. 77, 78, 79, ...) 
  # totalsize = randsizes[-1]
  # sizes = numpy.array(randsizes) / totalsize 

  # use relative training set size (e.g. 0, 0.1, 0.2, ..., 1.0) 
  sizes = numpy.arange(0, len(randsizes)) / (float(len(randsizes)) - 1)
  
  randarea = scipy.integrate.trapz(numpy.array(randaccs), sizes)
  selarea = scipy.integrate.trapz(numpy.array(selaccs), sizes)

  return selarea - randarea

#################################################################
#
#################################################################
if __name__ == "__main__":

  if len(sys.argv) == 2:
    dir = sys.argv[1]
  else:
    dir = DEFAULTDIR
    
  areas = []
  for file in os.listdir(dir):
    if file.endswith('.matlab'):
      a = area2('%s/%s' % (dir, file))
      areas.append(a)
      print file.split('.')[0], a

  print '\naverage:', sum(areas) / len(areas)
