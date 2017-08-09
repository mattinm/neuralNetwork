'''
This program compares the blob counts gotten from BlobCounter.cpp to true counts given by Marshall.
arg1 - output base name
arg2 - true csv
arg3... - other csvs

'''
import sys
import csv

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

def sumAbs(array):
	total = 0
	for i in array:
		total += abs(i)
	return total




# import matplotlib.pyplot as plt

#read in true vals
trueVals = {}
basename = sys.argv[1];
with open(sys.argv[2],'rU') as f:
	trueValReader = csv.reader(f)
	for row in trueValReader:
		trueVals[row[0]] = [int(row[1]),int(row[2])]

print "True vals size = " + str(len(trueVals))

#read in list of calculated vals
print "\n\n\n"
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
        print counts[-1].name + " - " + str(len(counts[-1].vals))


#compare and make output csvs
with open(basename+"_individuals.csv","wb") as oi, open(basename+"_aggregate.csv","wb") as oa:
	outIndiv = csv.writer(oi)
	outIndiv.writerow(["CNN/year","MSI","Calculated White","Actual White","Error White","Abs Error White", "Percent Error White","Calculated Blue","Actual Blue","Error Blue", "Abs Error Blue","Percent Error Blue","Total Error","Total Percent Error"])
	outAgg   = csv.writer(oa)
	outAgg.writerow(["CNN/year","Calculated White","Actual White","Error White","Abs Error White","Total Individual Abs Error White","Percent Error White","Calculated Blue","Actual Blue","Error Blue","Abs Error Blue","Total Individual Abs Error Blue","Percent Error Blue","Combined Error","Total Individual Abs Combined Error","Combined Percent Error"])
	for cnn in counts:
		totalCalc = [0,0]
		totalActual = [0,0]
		totalAbsError = [0,0]
		#individuals
		for key, calc in cnn.vals.iteritems():
			act = trueVals[key]

			calc.append(sum(calc)) #calc[-1] is the combined count
			act.append(sum(act))
			for i in range(0,2):
				totalCalc[i] += calc[i]
				totalActual[i] += act[i]
				totalAbsError[i] += abs(calc[i] - act[i]);
			perError = [0,0,0]
			for i in range(0,3):
				if(act[i] != 0):
					perError[i] = abs(calc[i]-act[i])/float(act[i])*100.
				elif(calc[i] == 0): #calc and actual are both 0
					perError[i] = 0
				else:
					perError[i] = 100 * calc[i];

			outIndiv.writerow([cnn.name,
				key,                   # msi
				calc[0],               # calc white
				act[0],                # actual white
				calc[0]-act[0],        # error white
				abs(calc[0]-act[0]),   # absolute value error white
				perError[0],           # % error white
				calc[1],               # calc blue
				act[1],                # actual blue
				calc[1]-act[1],        # error blue
				abs(calc[1]-act[1]),   # absolute value error blue
				perError[1],           # % error blue
				abs(calc[2] - act[2]), # absolute value total error
				perError[2]])          # % error combined based on abs val total error

		#aggregate
		totalCalc.append(sumAbs(totalCalc))
		totalActual.append(sumAbs(totalActual))
		totalAbsError.append(sum(totalAbsError))
		perError = [0,0,0]
		for i in range(0,len(totalCalc)):
			if(totalActual[i] != 0):
				perError[i] = abs(totalCalc[i]-totalActual[i])/float(totalActual[i])*100.
			elif totalCalc[i] == 0: #calc and actual are both 0
				perError[i] = 0
			else:
				perError[i] = 100 * totalCalc[i]
		outAgg.writerow([cnn.name,
			totalCalc[0],                      # calc white
			totalActual[0],                    # actual white
			totalActual[0]-totalCalc[0],       # error white
			abs(totalActual[0]-totalCalc[0]),  # abs error white
			totalAbsError[0],                  # total indiv absolute error white
			perError[0],                       # % error white
			totalCalc[1],                      # calc blue
			totalActual[1],                    # actual blue
			totalActual[1]-totalCalc[1],       # error blue
			abs(totalActual[1]-totalCalc[1]),  # abs error blue
			totalAbsError[1],                  # total indiv absolute error blue
			perError[1],                       # % error blue
			abs(totalCalc[2] - totalActual[2]),# abs error combined
			totalAbsError[2],                  # total indiv abs error combined
			perError[2]])                      # % error combined
