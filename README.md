# Pattern Analysis and Unsupervised Learning

### By Angel Villar-Corrales

This project contains the programming exercises of the Pattern Analysis course offered by the Computer Science departments at FAU, in addition to some other Unsupervised Learning algorithms and applications develop independently.


## Getting Started

To get the code, fork this repository or clone it using the following command:

>git clone https://github.com/angelvillar96/PatternAnalysis.git

### Prerequisites

To get the repository running, you will need several packages such as numpy, scipy, matplotlib or sklearn.

You can obtain them easily by installing the conda environment file included in the repository. To do so, run the following command from the Conda Command Window:

```shell
$ conda env create -f environment.yml
$ activate PatternAnalysis
```

*__Note__:* This step might take a few minutes

## Contents

This repository contains several folders with the exercises of the Pattern Analysis course offered at FAU, in addition some pattern analysis and unsupervised learning algorithms that I have developed independently.

The main goal of these is to illustrate different algorithms that I find interesting and to try them out in some real-world applications.

Each folder contains the folloring material:

  - **Jupyter Notebook**: notebook illustrating the algorithm and documenting the methods and the results

  - **Python file**: script containing the necessary methods and logic to implement the algorithms and the different applications

  - **Results (optional)**: folder containing images or precomputed results

  - **Docu (optional)**: folder containing research papers or the course handouts that further document the requirements and the scope of the code.

Here a present a brief overview of the exercises and algorithms, more information can be found in the Jupyter Notebook corresponding to the algorithms:

- **HMM**: Application of Hidden Markov Models to perform human signature verification in order to distinguish original from fake signatures.

- **K-Means**: Implemenation of the K-Means clustering algorithm, overview and explanation of the model selection problem, and implementation of Gap Statistics to find out the optimal K for the algorithm.

- **Mean-Shift**: Several implementations of the Mean Shift clustering algorithm and different applications of this method such as image denoising or image color segmentation.


- **Parzen-Window**: Implementation of the Parzen Window using different Kernels for probability distribution estimations.

- **Unsupervised to Supervised**: Here I illustrate how density estimation (inherently unsupervised) can be performed as a regression task (supervised learnning) using an auxiliar distribution and a Random Forest Regressor.

## Acknowledgment

Part of the algorithms were developed in close colaboration with Fabian Hübner.
