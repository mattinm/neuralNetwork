The programs in this repo are for training and running Convolutional Neural Networks over images. It uses OpenCV to get and write images and videos. It does its calculations in OpenCL. It was originally built for use with the Wildlife@Home project

The primary file format used to hold the training data is the IDX file format. For explanation see the bottom of this page http://yann.lecun.com/exdb/mnist/

IMPORTANT NOTE: The programs in this repository read in the data dimensions as Big Endian Int, while all data of any type is read in as Little Endian. This allows easier use on Intel machines for reading in the data while still being compatible with the MNIST dataset.

Wildlife_Loop_Train is written in Qt and has an interface where many parameters can be changed. It is the program that runs the feedback loop. It uses the other executables so they will have to be built in some directory.

Overview of source files:
Python:
	Blob_comparator.py - Wildlife@Home specific? Yes - Used to compare blob counts made by using BlobCounter.cpp over prediction images made by ConvNetFullImageDriverParallelCL to the actual blob counts given in a csv.

	Remove_artifacts.py - Wildlife@Home specific? Yes - Used to remove mosaiced images that had artifacting using simple algorithm

	SecondsConversion.py - Wildlife@Home specific? No - Used to convert seconds to HH:MM:SS.

C++:
	CNNtoObserver_comparator.cpp - Wildlife@Home specific? Yes - Compares prediction images to recorded user/expert observations. Can make IDX files of wrong data for retraining.

	CombineIDX_Thesis - Wildlife@Home specific? Yes - Combines two IDX files by appending on onto the other, allowing for specification of how much data percentage-wise is from each IDX.

	ConvNetCL.cpp - Wildlife@Home specific? No - Main cpp file for the neural network implementation. Made into a library. Uses Batch Normalization and Minibatch Gradient Descent. Also uses L2 Reg and Nesterov Momentum.

	ConvNetCommon.cpp - Wildlife@Home specific? Yes and no - Has some random helper functions used by multiple programs. Some are Wildlife@Home specific, others are not.

	ConvNetEvent.cpp - Wildlife@Home specific? Yes - Used with the Wildlife@Home video to easily handle individual Events and multiple Observations.

	ConvNetFullImageDriverParallelCL - Wildlife@Home specific? No - Breaks a large image up into a number of smaller sub-images that are run through the given CNN. Using CNN predictions a prediction image is made that shows what parts of the image were predicted to be what class.

	ConvNetSeam_ToIDX.cpp - Wildlife@Home specific? Yes - Brings in videos from Wildlife@Home database, seamcarves frames from the videos down to a fixed size, and make IDXs that have the video data and what class each video is.

	ConvNetSeam.cpp - Wildlife@Home specific? No - Deprecated - Old version of seamcarving code that is not thread safe.

	ConvNetSeamTrainerCL_idx - Wildlife@Home specific? No - Main CNN training program. Trains a CNN over given IDX files.

	ConvNetVideoDriverParallelCL.cpp - Wildlife@Home specific? No - Runs CNN over video similar to how ConvNetFullImageDriverParallelCL runs over images. Each frame is considered separately. Makes a prediction video.

	ConvNetVideoDriverParallelCL_multiNet.cpp - Wildlife@Home specific? No - Same as ConvNetVideoDriverParallelCL.cpp but allows an multiple CNNs to be used and their output averaged.

	DataVisualizer.cpp - Wildlife@Home specific? Kinda - Takes the data and label idxs and makes PNG images of the data. Multiple idx images are put on the same output image if the output image is big enough.

	IDX.cpp - Wildlife@Home specific? No - Class to handle IDX files. Can read/write IDX files. Can manipulate data. Automatically adjusts metadata on add/delete.

	kernelBuilder.cpp - Wildlife@Home specific? No - Builds an OpenCL kernel. If there are any errors it gives what output it can. If no errors, it states that the build was successful. Useful for debugging.

	Overlay.cpp - Wildlife@Home specific? Yes - Overlays CNN predictions on top of the original images.

	Seamcarver.cpp - Wildlife@Home specific? No - Class that does seamcarving. Do not use OpenCL implementations as they are not consistent.

	SeeUserObs.cpp - Wildlife@Home specific? Yes - Shows were user observations are for a specific MSI.

	SmoteIDX.cpp - Wildlife@Home specific? No - Uses SMOTE on images in an IDX to make more synthetic data. Works iffy for images tested.


Config file formats:
Deprecated - D1. For TrainingImageSplitterFileCreator
	- this creates the binary files that ConvNetTrainerCL and ConvNetContinuanceCL use.
	- has many options. Just type ./TrainingImageSplitterFileConverter after compiling to see options, or look in main.

	Format: (uint is unsigned int)
	subimageWidth<uint> subimageHeight<uint> subimageDepth<uint>
	sizeByte<int>

	#pound at beginning of line is a comment. Pound must be first character in line. Blank lines are fine too.

	#before the paths you can set the names of the classes with $set trueVal name
	$set trueVal<int> name<string>
	
	pathToFolder,trueVal<int>,stride<int>
	pathToFolder,trueVal<int>,stride<int>
	...

	Exp: (for 32x32x3 images with 2 possible classes)
	32 32 3
	1

	$set 0 Background
	$set 1 Dogs

	/Users/user/Images/class0,0,15
	/Users/user/Images/class1,1,30
	/Users/user/Images/moreClass0,0,20



	Notes: 
		- subimageDepth only works if it's 3 right now. It is an option because implementation for hyperspectral cameras may be added in the future. 
		- sizeByte is what it is stored as in the binary. For RGB images use 1. See comments in cpp file.
		- stride is how far to move over for the start of the next image. Does stride in both x and y directions. Larger stride means fewer subimages with an exponentional (stride^2) decrease. If no stride is put, it defaults to 1.