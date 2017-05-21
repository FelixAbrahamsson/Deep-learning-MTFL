To run the code, make sure all the code is in it's own folder, and put the MTFL data set (found here http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html) in a separate folder named MTFL, as a child directory to the parent directory of the code directory.

Make sure the "resize.py" script is run before running the main file, it will resize the images to fit the network and apply data augmentation.

The network can be run either from the "main.py" file using command line arguments, or from the "main_simple.py" file where the networks are configured from within the script itself.