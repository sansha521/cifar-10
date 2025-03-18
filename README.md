# Team YAN: Deep Learning CS 6953 / ECE 7123 2025 Spring

This is the project repository for the team YAN (Yumiko, Athul & Nidhi). 


# Install Libraries.

We have only used minimal libraries. The libraries are mentioned in the `requirements.txt` file. 

# Training. 

We had used Jupyter notebook in the HPC burst node to train. However, we have modularized the code to run in a python file. 
The final model we have used is ResNetSE. It is available in the directory `models/resnet_se.py`. 

> Please note that you have to change the parameters mentioned in the `config.ini` file.

Also, do not change the path of the pickle file used for the final prediction. It is hard coded as we do not the file or the file path to be modified. 
