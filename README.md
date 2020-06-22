#######################FOR TRAINING THE NETWORK YOURSELF################
Open the Preprocess file , Enter the 3 Directory , where to load the chinese file , where to store the InputFile , Where to store the Label File 

This step needs to be done only if you want to retrain the whole model , Change the variable "Training" at line 23 to True

Step #2 
If you want to Train the Network ,Edit the Directories at top , these 3 variables needs to be set . Once done run python3 "train.py"


#######################IF You want to Just predict or score################


Sample Way to run the Predict


"python3 predict.py "/home/baqai/Desktop/code/public_homework_1/icwb2-data/training/as_Input1.utf8" "/home/baqai/Desktop/code/public_homework_1/sample_files/aa.txt" "/home/baqai/Desktop/code/public_homework_1/resources"

python3 predict.py INPUTSFILE OUTPUTFILEWITHLABELS RESOURCESFOLDER

Python3 score.py TRUELABEL.txt OUTPUTFILEWITHLABELS