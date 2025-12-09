# If Glare Doesn’t Fool Humans, It Shouldn’t Fool Your System: Reliability Challenges in Traffic Sign Classification

> Abstract


Autonomous vehicles rely on deep learning based per-
ception systems to interpret traffic signs and support safe
driving decisions. Although previous studies have exam-
ined environmental factors such as rain, shadows, and par-
tial occlusions, the influence of sun glare has received far
less attention despite its frequent presence in real-world
driving. This work provides a comprehensive analysis of
how natural and synthetically generated sun glare affects
traffic sign classification. Using a controlled glare gen-
eration workflow, we create datasets with multiple glare
positions and intensity levels to evaluate the robustness of
several architectures, including a custom CNN, VGG16,
and ResNet variants. Our experiments demonstrate that
sun glare causes substantial performance degradation and
that certain traffic sign categories are more sensitive to it
than others. Results indicate that sun glare significantly
decreases the traffic sign classification accuracy from 97%
to 22%. We also found that digit-based signs, such as
speed limit 45, 65, are the most vulnerable. Furthermore,
we found that the center of the traffic sign is more vul-
nerable to sun glare, which decreases the test accuracy by
80%.



# Prerequisites
## This code was primarily run on Ubuntu 22.04.5. However, nothing is depending on specific version of Ubuntu. You are required to have [virtual environment](https://docs.python.org/3/library/venv.html) in your system.

### How to installed virtual environment in your system?

``` bash
python -m venv .venv          # create
source .venv/bin/activate     # activate (Linux/macOS)
.venv\Scripts\activate        # activate (Windows)
pip install -r requirements.txt

```

### How to get the Lllma4 API

Check this link: [Llama4](https://medium.com/data-science-in-your-pocket/how-to-use-meta-llama4-for-free-da46c30aa32c)


> add_glare.py

This is the file I use to add the artificial sun glare on the traffic sign in different position of the sign from top to bottom to left to right towards center. You just need to change the value of the `position`. For example, `position="center"`.

> cnn.ipynb

This file is the custom made cnn model. You just need to change different dataset directory to perform the cnn model.


> vgg16.ipynb

This file contains the all the five models, what you need to do is just change the dataset from normal to glare. You can also change the folder name for saving the figure for normal and glare.



## Repository Structure

This repository is containes several folders, each serving a specific purpose in our study. Below is a table detailing each folder and it's contents.

| Folder Name              | Description                                                                                                  | README Link                                   |
|--------------------------|--------------------------------------------------------------------------------------------------------------|-----------------------------------------------|
| `1_Datasets`             | Contains the datasets, each with images, human-annotated ground truth with CSV files, and glare induced dataset in 5 different position.       | [README](./1_Datasets/README.md)              |
| `2_GenerateDescriptions` | Contains the prompt for VLM or Llama4 model.                       | [README](./2_GenerateDescriptions/README.md)  |
| `3_GenerateResults`      | Code for analyzing study results and data presentation.                                                      | [README](./3_GenerateResults/README.md)       |
| `Results`      | Contains all the CSV and figures files.                                                      | [README](./Results/README.md)       |