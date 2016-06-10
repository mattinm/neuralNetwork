import sys
import csv
import copy
sys.path.insert(1, "/usr/local/lib/python2.7/site-packages")
import numpy as np
import scipy
#print scipy.__version__
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

print(sys.argv[1])
files = open(sys.argv[1], 'r')

time = []
red = []

count = 0
with open(sys.argv[1], 'r') as f:
	reader = csv.reader(f)
	for row in reader:
		if count % 10 == 0:
			red.append(float(row[0]))
			time.append(float(row[1]))
		count += 1

print len(red)
print len(time)

fil = copy.deepcopy(red)
for i in range(0, len(fil)):
	fil[i] += 1000000

#apply the filter to red
fil = savgol_filter(fil, 7, )

for i in range(0,len(fil)):
	print str(red[i]) + " - " + str(fil[i])

plt.plot(time, red)
plt.plot(time, fil)
plt.show();