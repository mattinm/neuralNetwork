'''
This program compares the blob counts gotten from BlobCounter.cpp to true counts given by Marshall.
arg1 - true csv

'''
class CountClass:
	def __init__(__self__):
		pass

def getMSI(filename):
    if(filename.isdigit()):
        return filename;
	startMSIIndex = filename.find("msi");
	nextUnderscore = filename.find("_",startMSIIndex);
	if(nextUnderscore == -1):
		nextUnderscore = filename.find(".",startMSIIndex);
	return filename[startMSIIndex+3:nextUnderscore]

# def sum(array):
# 	total = 0
# 	for i in array:
# 		total += i
# 	return total



import sys
import csv
# import matplotlib.pyplot as plt

#read in true vals
trueVals = {}
removed = 0
msis_removed = []
with open(sys.argv[1],'rU') as f:
	trueValReader = csv.reader(f)
	for row in trueValReader:
		msi = int(row[0])
		if(msi > 2000 or msi < 1300 or (msi % 10 != 0 and msi % 100 < 90)):
			trueVals[row[0]] = [int(row[1]),int(row[2])]
		else:
			removed += 1
			msis_removed.append(msi)

print "Removed " + str(removed) + " images\n"
for msi in msis_removed:
	print "\t" + str(msi) + "\n"


outname = sys.argv[1][0:sys.argv[1].rfind(".")] + "_no_artifact.csv"


#compare and make output csvs
with open(outname,"wb") as oi:
	out = csv.writer(oi)
	for key, val in trueVals.iteritems():
		out.writerow([key,val[0],val[1]])
