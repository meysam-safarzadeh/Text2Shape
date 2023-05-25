## Dependencies
The project requires the following dependencies to be installed:

pyvista==0.39.0
torch==2.0.0
numpy==1.23.5
pandas==1.5.3
spacy==3.3.1
language_tool_python==2.5.1
pynrrd==1.0.0
matplotlib==3.7.0

Please ensure that these dependencies are installed before running the code. The code has been implemented and tested using Python 3.9.16 with conda environment management.

Code Attribution
Meysam Safarzadeh:
train.py
model.py
custom_test.py
Tested and debugged the final code

Shaurya Rawat:
Text description preprocessing
encoder.py

Nelson Evbarunegbe:
3D shape preprocessing
generator.py


## Instructions
Installation
To install the required dependencies, follow these steps:

Set up a virtual environment (recommended).

Install the dependencies using the following command:

pip install -r requirements.txt


## Training
To train the model, perform the following steps:
having a GPU with at least 6 GB RAM is required to train. The model is trained and tested with Titan Xp with 12 GB RAM and python 3.9.16 on Ubuntu OS with 64 GB RAM.

1. first you need to download the dataset from: http://text2shape.stanford.edu/dataset/shapenet/nrrd_256_filter_div_32_solid.zip

2. you need to change the file_paths in data/captions.tablechair.csv so that it can be matched with the path of the downloaded dataset.

3. Run the training script:
python train.py


## Costum Testing
To test the trained model on your arbitrary text prompts, follow these steps:

edit text_list in the custom_test.py
Run the testing script:
python custom_test.py

