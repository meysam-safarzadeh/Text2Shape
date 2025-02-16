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

## Implementation

The implementation of this method involves three interconnected modules that operate under full supervision. As shown in Figure 1, modules include text preprocessing, BERT-based text encoding, a convolutional neural network (CNN) based text encoder architecture, and a shape generation architecture. The model leverages Mean Squared Error (MSE) loss and the Adam optimizer for training.

![image](https://github.com/meysam-safarzadeh/Text2Shape/assets/51737180/84c7abb2-0aa7-4475-87e0-447bee94daa1 "Summary of the model")

*Figure 1 Summary of the model*


### 1 Text Preprocessing
#### 1.1 Lowercasing
- **Purpose:** Ensuring consistency by converting natural language descriptions to lowercase.

#### 1.2 Tokenization and Lemmatization
- **Tools:** Utilizing Spacy pipeline for tokenization and lemmatization.

#### 1.3 Spell Correction
- **Method:** Employing LanguageTool, which integrates hunspell, grammar rules, and custom spelling corrections, specifically for the ShapeNet dataset.

#### 1.4 Description Length Filtering
- **Criteria:** Excluding descriptions exceeding 64 tokens from further processing.

### 2 BERT-based Text Encoding
#### 2.1 Base-Uncased Variant
- **Application:** Obtaining informative embeddings of textual inputs.

#### 2.2 Frozen BERT Model
- **Strategy:** Keeping the BERT model frozen during training, differentiating from the text2shape approach.

### 3 Text Encoder Architecture
#### 3.1 CNN Architecture
- 

#### 3.2 Encoding Process
- **Function:** Encoding each embedded text into a 128-dimensional condensed representation.

#### 3.3 Architecture Details
- **Components:** Involves convolutional layers, a GRU (Gated Recurrent Unit) with 256 units, ReLU activation functions, batch normalization, and L2 regularization.
- **Reference:** See Table 1 for detailed architecture specifications.
  ![image](https://github.com/meysam-safarzadeh/Text2Shape/assets/51737180/e36f5b36-fa67-4191-8786-2951d5ef1fbf "Model architecture")

  *Table 1 Model architecture*


### 4 Shape Generation Architecture
#### 4.1 Fractionally-strided Convolutions
- **Usage:** Employed for upsampling within the shape generation architecture.

#### 4.2 Activation Functions
- **Application:** ReLU functions for all layers except the final one, which uses a sigmoid activation function.

#### 4.3 Color Prediction
- **Capability:** Designed to predict both voxel existence and color, with plans for future iterations to separate these outputs.

### 5 Loss and Optimizer
#### 5.1 Mean Squared Error (MSE) Loss
- **Role:** Measuring the discrepancy between predicted outputs and ground truth targets, aiding in generating accurate voxel representations of 3D shapes.

#### 5.2 Adam Optimizer
- **Choice:** Utilized for optimizing model parameters, combining adaptive learning rates with momentum-based updates, ideal for high-dimensional parameter spaces.


## Results


The first observation from the training/validation loss is
that the model is performing well but has not reached its
optimal performance, given that we only use 10% of the whole dataset due to hardware limitations. This is evident from Figure 2, as the
overall trend in the loss shows decrement with each epoch.
Here, the training loss falls rapidly at first before obtaining
a gradual rate. Whereas, the validation loss shows a volatile
progress with various peaks and valleys in the graph
however the general trend shows an improved
performance. On experimenting with the test set, a loss of
0.0472 was observed.
  ![image](https://github.com/meysam-safarzadeh/Text2Shape/blob/main/results/plot.png "Train/val loss")


Figure 2: A graph showing the change in training and
validation loss with each epoch


As for the generated shapes, several examples are
illustrated in the figures. Figure 3 demonstrates the result of
a text prompt describing a “simple sofa with 4 legs.” The
generated shape shows 4 legs and the
presence of a seat back with a round shape as it is in a typical sofa. However, it is apparent that the
model has not fully converged, as the output lacks complete
accuracy.

![image](https://github.com/meysam-safarzadeh/Text2Shape/assets/51737180/60ec56ac-c3e2-43ee-b764-1ec9162d3cf9 "")

Figure 3 Prompt: "Simple sofa with 4 legs."

Figure 4 showcases another
example, where the prompt was “seat with back support”
for a chair. It is noticeable that one of the legs is
incomplete, and the shape is less rounded compared to the sofa in Figure 3.
![image](https://github.com/meysam-safarzadeh/Text2Shape/assets/51737180/5d66f469-f8b3-453b-b132-58c46bcdfdce)

*Figure 4 Prompt: "seat with back support"*

Figure 5 displays the generated shape for the prompt “big
chair with armrest” in real colors. As can be seen, armrests are generated and are acceptable to some extent. Despite efforts to
represent colors accurately, the model struggles to
understand and generate the desired color prompts
ineffectively, resulting in predominantly brownish shades.

![image](https://github.com/meysam-safarzadeh/Text2Shape/assets/51737180/8bbb5a8a-9ac3-458b-8554-bed7f22fb07d)

*Figure 5 Prompt: "big chair with armrest"*


Lastly, Figure 6 demonstrates the generated output for the
prompt “big table desk”, indicating that the model faces
challenges in generating accurate table structures. This
suggests that additional training is required to improve the
model’s understanding that will result in a better generation
of tables and other complex objects.

![image](https://github.com/meysam-safarzadeh/Text2Shape/assets/51737180/88221b79-c300-4865-b813-dff0d9f8caf5)

*Figure 6 Prompt: "big table desk"*


## Discussion and Conclusion

While the model can differentiate between high-level
features such as chairs and desks, it falls short when it
comes to low-level features such as differentiating between
types of chairs like three-leg chair and four-leg chair.
Another limitation observed is that the model tends to
group similar items like chairs, sofas, and seats together.
This happens maybe because of several similar properties
as even in real life, some chairs and sofas look very similar.
One of the ways to overcome these limitations is to use a
larger and more diverse dataset that includes synthesizing
new samples from current ones. A much wider range of
examples will help in learning the subtle differences.
Another way is to extend the duration of training from the
current one. Training for a longer time with more epochs
will enhance the model's capabilities and the results
produced will be better than the previous results.
This field has a wide scope in several industries such as
computer-aided design (CAD), virtual reality (VR), and
gaming. It can help in improving the semantic
understanding of text data for higher precision in results.
The field can also be used in the future to offer more
fine-grained control over the creation of shapes. This
includes modifying features such as orientation, size, and
material properties. It can help in enabling user interactions
for specific outputs and help in refining/iterating
over-generated shapes. The usage can be domain-specific
like Industrial design, product development, or
architecture. And finally, it can be used for data
augmentation and synthesis. This can be achieved by
various techniques relevant to 3D shapes.


