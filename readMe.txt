The programs in this repo are for training and running Convolutional Neural Networks over images. It uses OpenCV to get and write images and videos. It does its calculations in OpenCL. The general workflow if you want to use it is this:

1. Use TrainingImageSplitterFileConverter to make a binary file out of all your images. It will take full sized images and make subimages from them and put them all into a binary file.
1a. (optional) Use TrainingImageSplitterFileConverter to make a binary file for your test set that can be used to evaluate training.

2. Change the network dimensions and layout in ConvNetTrainerCL and recompile. Then use ConvTrainerCL to train the network. If you made a test set, you can add it in here also, but don't worry, it won't train on the test set. 
2a. (optional) Use ConvNetTester to see on what classes of images it normally misclassifies.
2b. (optional) Use ConvNetContinuanceCL to do more training on already trained networks. Can be useful for training extra on types of images it tends to misclassify.

3. Run your newly trained CNN over full images or videos using ConvNetFullImageDriverCL or ConvNetVideoDriverCL. For extra speed use the parallel versions which use all OpenCL capable devices on your machine that support doubles.
3a. (optional) Use the BOINC or MPI versions of the video driver for large videos or a large amount of videos.
