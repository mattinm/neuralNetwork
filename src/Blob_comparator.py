'''
This program compares the blob counts gotten from BlobCounter.cpp to true counts given by Marshall.
arg1 - true csv

'''
class CountClass:
	def __init__(__self__):
		pass

def getMSI(filename):
	startMSIIndex = filename.find("msi");
	nextUnderscore = filename.find("_",startMSIIndex);
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
with open(sys.argv[1],'rU') as f:
	trueValReader = csv.reader(f)
	for row in trueValReader:
		trueVals[row[0]] = [int(row[1]),int(row[2])]

#read in list of calculated vals
counts = []
for i in range(2,len(sys.argv)):
	with open(sys.argv[i],'rU') as f:
		reader = csv.reader(f)
		counts.append(CountClass())
		counts[-1].name = sys.argv[i]
		counts[-1].vals = {}
		for row in reader:
			msi = getMSI(row[0])
			if(msi in trueVals.keys()):
				counts[-1].vals[msi] = [int(row[1]),int(row[2])]


#compare and make output csvs
with open("Blob_compare_errors_individuals.csv","wb") as oi, open("Blob_compare_errors_aggregate.csv","wb") as oa:
	outIndiv = csv.writer(oi)
	outIndiv.writerow(["CNN/year","MSI","Calculated White","Actual White","Error White","Percent Error White","Calculated Blue","Actual Blue","Error Blue","Percent Error Blue","Total Error","Total Percent Error"])
	outAgg   = csv.writer(oa)
	outAgg.writerow(["CNN/year","Calculated White","Actual White","Error White","Percent Error White","Calculated Blue","Actual Blue","Error Blue","Percent Error Blue","Total Error","Total Percent Error"])
	for cnn in counts:
		totalCalc = [0,0]
		totalActual = [0,0]
		for key, calc in cnn.vals.iteritems():
			act = trueVals[key]
			# combCalc = sum(calc)
			# combAct = sum(act)
			calc.append(sum(calc)) #calc[-1] is the combined count
			act.append(sum(act))
			for i in range(0,2):
				totalCalc[i] += calc[i]
				totalActual[i] += act[i]
			perError = [0,0,0]
			for i in range(0,3):
				if(act[i] != 0):
					perError[i] = abs(calc[i]-act[i])/float(act[i])*100.
				elif(calc[i] == 0): #calc and actual are both 0
					perError[i] = 0
				else:
					perError[i] = 100
			outIndiv.writerow([cnn.name,key,
				calc[0],act[0],abs(calc[0]-act[0]),perError[0], #white
				calc[1],act[1],abs(calc[1]-act[1]),perError[1], #blue
				abs(calc[2] - act[2]),perError[2]])     #combined
		# combCalc = sum(totalCalc)
		# combAct = sum(totalActual)
		totalCalc.append(sum(totalCalc))
		totalActual.append(sum(totalActual))
		perError = [0,0,0]
		for i in range(0,len(totalCalc)):
			if(totalActual[i] != 0):
				perError[i] = abs(totalActual[i]-totalCalc[i])/float(totalActual[i])*100.
			elif totalCalc[i] == 0: #calc and actual are both 0
				perError[i] = 0
			else:
				perError[i] = 100
		outAgg.writerow([cnn.name,
			totalCalc[0],totalActual[0],abs(totalActual[0]-totalCalc[0]),perError[0], #white
			totalCalc[1],totalActual[1],abs(totalActual[1]-totalCalc[1]),perError[1],
			abs(totalCalc[2] - totalActual[2]),perError[2]])
