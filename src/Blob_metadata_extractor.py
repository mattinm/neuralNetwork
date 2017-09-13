'''
This program gets metadata information about how each CNN did on the data, much of it aggregating per image stats.

Call: python Blob_metadata_extractor.py blob_individual_data1 blob_individual_data2 ...
	blob_individual_data(s) are an output csv from Blob_comparator.py with the ending "_individuals.csv"

List of information retrieved on per CNN basis:
 1. Number of images with 0 error
 2. Number of images with > 0 error
 3. Number of images that 25%, 50%, 75%, and 100% of error is from. 
 4. Percent of images that 25%, 50%, 75%, and 100% of error is from. 
 5. Sum guesses from individual images;
 	a. Total false positives
 	b. Total false negatives
'''

import sys
import csv

'''
Read in data
'''
cnns = {} #key: CNN name as string. value: list of lists. Inner lists are lines from csv minus the cnn_name
          #inner list: 0)msi, 1)calc white, 2)actual white, 3)error white, 4)abs error white, 5)%error white, 
          #       6)calc blue, 7)actual blue, 8)error blue, 9)abs error blue, 10)%error blue, 11)abs total error, 12)%error total
ABS_WHITE_ERROR = 4
for i in range(1, len(sys.argv)):
	with open(sys.argv[i],'rU') as f:
		reader = csv.reader(f)
		firstRow = True
		for row in reader:
			if(firstRow): #ignore header
				firstRow = False
				continue
			key = row[0];
			if key == "":
				continue
			float_row = row[1:]
			for f in range(len(float_row)):
				float_row[f] = float(float_row[f])
			if key in cnns:
				cnns[key].append(float_row)
			else:
				cnns[key] = [float_row];


'''
Per CNN get number of images with 0 error and > 0 error
'''
cnn_image_errors = {} # dict of tuples in format (im w/ 0 error, im w/ > 0 error, total ims). key cnn_name
cnn_num_files_per_percent_error = {} # dict of tuples in format (num ims for 25% error, 50%, 75%, 100%). key cnn_name
cnn_percent_files_per_percent_error = {} # dict of tuples in format (% of ims for 25% error, 50%, 75%, 100%). key cnn_name


#for every cnn, sort the outer list descending by the abs error white in the inner list
keys = cnns.keys()
keys.sort()
# for cnn,infolist in cnns.iteritems():
for cnn in keys:
	infolist = sorted(cnns[cnn],key=lambda info: info[ABS_WHITE_ERROR], reverse=True)

	# get info for 1 & 2
	total_images = len(infolist)
	start_of_0_error = total_images # when we start getting 0 errors
	total_error = 0.0
	for i in range(0,total_images):
		if(infolist[i][ABS_WHITE_ERROR] == 0):
			start_of_0_error = i
			break
		total_error += infolist[i][ABS_WHITE_ERROR];
	num_0_error = total_images - start_of_0_error
	num_gr_0_error = total_images - num_0_error
	cnn_image_errors[cnn] = (num_0_error, num_gr_0_error, total_images)

	print cnn + ":"
	print "\tNum 0 error = " + str(num_0_error)
	print "\tNum > 0 error = " + str(num_gr_0_error)
	print "\tTotal images = " + str(total_images)
	print ""

	#get info for 3, 4, 5 (a & b)
	if(start_of_0_error != 0): #if everything is perfect then we don't need to do this
		curError = 0;
		marksHit = 0
		n100 = start_of_0_error;
		print "\tTotal Error: " + str(total_error)
		for i in range(0,start_of_0_error):
			curError += infolist[i][ABS_WHITE_ERROR]
			if marksHit < 1 and curError / total_error >= .25:
				n25 = i + 1 # if it happens on first image, we don't want the division to result in 0
				marksHit += 1
			if marksHit < 2 and curError / total_error >= .50:
				n50 = i + 1
				marksHit += 1
			if marksHit < 3 and curError / total_error >= .75:
				n75 = i + 1
				break;
		cnn_num_files_per_percent_error[cnn] = (n25,n50,n75,n100)
		cnn_percent_files_per_percent_error[cnn] = (float(n25)/total_images * 100.0,float(n50)/total_images * 100.0,float(n75)/total_images * 100.0,float(n100)/total_images * 100.0)

		print "\tn25  = " + str(n25)
		print "\tn50  = " + str(n50)
		print "\tn75  = " + str(n75)
		print "\tn100 = " + str(n100)
		print ""
		print "\tp25  = " + str(cnn_percent_files_per_percent_error[cnn][0])
		print "\tp50  = " + str(cnn_percent_files_per_percent_error[cnn][1])
		print "\tp75  = " + str(cnn_percent_files_per_percent_error[cnn][2])
		print "\tp100 = " + str(cnn_percent_files_per_percent_error[cnn][3])