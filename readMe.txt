This repository is for tools useful for Convolutional Neural Networks for Image Processing.

This is how to use the files.

---------------------------------------------------------------------------------------------------------------

ImageSplitter splits an image into 32x32 subimages. It starts in the top left of the image and goes across and down by an optional stride value that defaults to 1. The new images are named based on the baseOutputName argument. They will be named baseOutputName0.ext, baseOutputName1.ext, etc, with the .ext being the inputted extenstion. Right now all images in the folder or path must be of type .jpg, .jpeg, or .png.

	use format:
	./ImageSplitter imageOrFolderPath outputFolderPath baseOutputName outputExtension (stride=1)

---------------------------------------------------------------------------------------------------------------

ConvNetTest requires the name of a trainingImagesConfig file as input. For this, all files of a different class must be in a different folder. The format for the file goes as follows:

folderPathContainingImagesOfClass0,trueValOfClass0
folderPathContainingImagesOfClass1,trueValOfClass1
folderPathContainingImagesOfClass2,trueValOfClass2

The trueVals are ints ranging from 0 to numClasses with each class having a unique number. Multiple folders can contain classes of the same type, but a single folder can only contain classes of one type.

Exp:

dogs,0
cats,1
moreDogs,0