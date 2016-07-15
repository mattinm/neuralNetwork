The programs in this repo are for training and running Convolutional Neural Networks over images. It uses OpenCV to get and write images and videos. It does its calculations in OpenCL. The general workflow if you want to use it follows.

Workflow:
	0. Get training data images and put them into folders sorted by class. i.e. put all the dogs in one folder and all the cats in another. 
	0a. Clean all your training data. If you are going to use 32x32 as the input for your CNN make sure your  training examples don't have any areas that are 32x32 or larger that don't contain at least part of that class. i.e. if your dog is in a house make sure there aren't any 32x32 parts of the image that contains only the floor and/or walls.
	0b. (optional). Do 0 and 0a for a set of testing images. Or pull out some of your training images for testing.

	1. Use TrainingImageSplitterFileConverter to make a binary file out of all your images. It will take full sized images and make subimages from them and put them all into a binary file.
	1a. (optional) Use TrainingImageSplitterFileConverter to make a binary file for your test set that can be used to evaluate training.

	2. Create a CNN config file (see CNN_config_example.txt). Then use ConvTrainerCL to train the network. If you made a test set, you can add it in here also, but don't worry, it won't train on the test set. 
	2a. (optional) Use ConvNetTester to see on what classes of images it normally misclassifies.
	2b. (optional) Use ConvNetContinuanceCL to do more training on already trained networks. Can be useful for training extra on types of images it tends to misclassify.

	3. Run your newly trained CNN over full images or videos using ConvNetFullImageDriverCL or ConvNetVideoDriverCL. For extra speed use the parallel versions which use all OpenCL capable devices on your machine that support doubles.
	3a. (optional) Use the BOINC or MPI versions of the video driver for large videos or a large amount of videos.


Config file formats:
A. TrainingImageSplitterFileCreator
	- this creates the binary files that ConvNetTrainerCL and ConvNetContinuanceCL use.
	- has many options. Just type ./TrainingImageSplitterFileConverter after compiling to see options, or look in main.

	Format: (uint is unsigned int)
	subimageWidth<uint> subimageHeight<uint> subimageDepth<uint>
	sizeByte<int>
	#pound at beginning of line is a comment. Pound must be first character in line. Blank lines are fine too.
	pathToFolder,trueVal<int>,stride<int>
	pathToFolder,trueVal<int>,stride<int>
	...

	Exp: (for 32x32x3 images with 2 possible classes)
	32 32 3
	1
	/Users/user/Images/class0,0,15
	/Users/user/Images/class1,1,30
	/Users/user/Images/moreClass0,0,20



	Notes: 
		- subimageDepth only works if it's 3 right now. It is an option because implementation for hyperspectral cameras may be added in the future. 
		- sizeByte is what it is stored as in the binary. For RGB images use 1. See comments in cpp file.
		- stride is how far to move over for the start of the next image. Does stride in both x and y directions. Larger stride means fewer subimages with an exponentional (stride^2) decrease. If no stride is put, it defaults to 1.