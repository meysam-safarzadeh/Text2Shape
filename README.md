# 3D Shape Generator using text prompts (Text2Shape supervised learning)

This project is a machine learning model that performs 3D shape (voxelization) generation. It takes text prompts and produces 3D shapes. The model is implemented using supervised learning and has been tested on pytorch.

## Dependencies

The project requires the following dependencies to be installed:

- pyvista==0.39.0
- torch==2.0.0
- numpy==1.23.5
- pandas==1.5.3
- spacy==3.3.1
- language_tool_python==2.5.1
- pynrrd==1.0.0
- matplotlib==3.7.0

Please ensure that these dependencies are installed before running the code. The code has been implemented and tested using Python 3.9.16 with conda environment management.

## Code Attribution

All codes in this project have been written by Meysam Safarzadeh.

## Instructions

### Installation

To install the required dependencies, follow these steps:

1. Set up a virtual environment (recommended).
2. Install the dependencies using the following command:

   ```shell
   pip install -r requirements.txt
   ```

### Training

To train the model, perform the following steps:

Note: Training the model requires a GPU with at least 6 GB RAM. The model has been trained and tested with Titan Xp with 12 GB RAM and Python 3.9.16 on Ubuntu OS with 64 GB RAM.

1. Download the dataset from [dataset download link](http://text2shape.stanford.edu/dataset/shapenet/nrrd_256_filter_div_32_solid.zip).
2. Update the file paths in `data/captions.tablechair.csv` to match the path of the downloaded dataset.
3. Run the training script:

   ```shell
   python train.py
   ```

### Custom Testing

To test the trained model on your arbitrary text prompts, follow these steps:

1. Edit the `text_list` variable in `custom_test.py` to include your desired text prompts.
2. Run the testing script:

   ```shell
   python custom_test.py
   ```

Please refer to the code comments and documentation for more details on how to use the project.
