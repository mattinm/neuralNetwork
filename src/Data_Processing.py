import sys
import csv
import copy
sys.path.insert(1, "/usr/local/lib/python2.7/site-packages")
import numpy as np
import scipy
import matplotlib.pyplot as plt

#defines
UNKNOWN = -1
OUT_OF_FRAME = 0
ON_GROUND = 1
FLYING = 2


print(sys.argv[1])
#files = open(sys.argv[1], 'r')

time = []
red = []

count = 0
with open(sys.argv[1], 'rU') as f:
	reader = csv.reader(f)
	for row in reader:
		if count % 10 == 0:
			#print row[0]
			red.append(float(row[0]))
			time.append(float(row[1]))
		count += 1


mean = np.mean(red)

changeNeeded = .5

len_minus_10 = len(red)-10;
changes = [] # holds the indexes for all the changes
for i in range(0, len(red)):
	#if(red[i] > mean):
		if(i > 10): #avg the 10 prev
			submean = np.mean(red[i-10:i])
			if(abs(red[i]-submean) > submean * changeNeeded): # if the point is more than a 40% change in the submean
				changes.append(i)
		if(i < len_minus_10):
			submean = np.mean(red[i:i+10])
			if(abs(red[i]-submean) > submean * changeNeeded): # if the point is more than a 40% change in the submean
				changes.append(i)

# after all changes found. make the graph straight lines on the plateaus
plateau = []
#make plateau right size
for q in range(0, len(red)):
	plateau.append(0)

changes.append(len(red)); #this is so the for loop gets the end part
changes = np.unique(changes) #remove duplicates and order
print changes
flats = []
flatClassification = []
#first part
submean = np.mean(red[0:changes[0]])
flats.append(submean)
flatClassification.append(UNKNOWN)
for j in range(changes[0]):
	plateau[j] = submean;
#rest of it
for i in range(0, len(changes)-1):
	submean = np.mean(red[changes[i]+1:changes[i+1]])
	flats.append(submean)
	flatClassification.append(UNKNOWN)
	plateau[changes[i]] = red[changes[i]]
	for j in range(changes[i]+1,changes[i+1]):
		plateau[j] = submean


#find lowest flat. this is baseline for OUT_OF_FRAME

minVal = np.min(flats)
for i in range(0, len(flats)):
	if(flats[i] < 1.2 * minVal):
		flatClassification[i] = OUT_OF_FRAME
	else:
		flatClassification[i] = ON_GROUND

print flatClassification
print flats
plt.plot(time, red)
plt.xlabel("Time (seconds)")
plt.plot(time, plateau, linewidth=2.0)
for i in range(0,len(flats)):
	if(i == 0):
		if(flatClassification[0] == ON_GROUND):
			plt.axvspan(0, changes[0]-1, color='red', alpha=0.2, lw=0)
		elif(flatClassification[0] == OUT_OF_FRAME):
			plt.axvspan(0, changes[0]-1, color='purple', alpha=0.2, lw=0)
	else:
		if(flatClassification[i] == ON_GROUND):
			plt.axvspan(changes[i-1],changes[i]-1,color='red',alpha=0.2,lw=0)
		elif(flatClassification[i] == OUT_OF_FRAME):
			plt.axvspan(changes[i-1],changes[i]-1,color='purple',alpha=0.2,lw=0)

plt.show();