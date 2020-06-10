### TITLE - <br>
Detection of Diabetic Retinopathy using Convolutional Neural Network

### AUTHORS - 
	1. Razat Agarwal
	2. Noopur Gautam
	3. Aditya Mahamuni
	4. Piyush Awachar
	
Validation Data can be downloaded from this drive link -> 
	
### Steps to use the CLI - 

1. Clone the repository from github
2. Install the dependencies from the requirements.txt document.
3. Run pre_processing.py on the new image for blood vessel segmentation.
4. Open main.py and change the variables according to the working directory on line 137 - 141.
5. Run main.py with the desired argument.

### Arguments for command line - 
-h -> To get help for the CLI
-a, --all -> Run the analysis on all the validation images and print CONFUSION MATRIX and ACCURACY
-i, --image -> Run the analysis on one image, name to be given as the second argument.

### Example - 
python3.6 main.py -i image.png
