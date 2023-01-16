## ISTA

ISTA is a tool for automatic test case generation and test case optimization based on neuronal coverage analysis. ISTA can improve the test adequacy of the intelligent system while expanding the dataset(text data type and image data type). 

In summary, the ISTA tool provides the following benefits:

- **Various data types.** ISTA supports not only image data type datasets but also text data type datasets.
- **Flexibility and extensibility.** With a modular approach to integration, new technologies and methods can be added to the ISTA with less effort.
- **Interaction friendly.** ISTA has a user-friendly interface with a project directory area, a function operation display area, and a log information display area.
- **Data visualisation.** The results of the experiments can be presented in a visual format.
- **Results report.** ISTA can automatically generate test reports with key information from the testing and evaluation process.

## Components

ISTA is implemented based on the PyQt5 and the TensorFlow framework, and the architecture is shown in the figure below.

<img width="719" alt="overview" src="https://user-images.githubusercontent.com/100778073/212594386-bef61398-5913-45b6-ade3-26473d3e4415.png">


There are 5 layers, i.e., the expression layer, the computation layer, the generation layer, the optimization layer, and the platform layer. ISTA is designed to be flexible and extensible, i.e., users can follow the original architecture of ISTA and add new technologies and methods to specific layers according to new requirements.

- **The expression layer** is used to create new projects, import models, and import datasets.
- **The computation layer** uses the coverage criteria applied to deep neural networks as test adequacy criteria. 
- **The generation layer** is used to generate test cases that can effectively address the problem of small size and a single class of deep neural network datasets while improving test adequacy.
- **The optimization layer** is used to optimize test cases.  The goal of the optimization layer is to meet the given adequacy testing requirements with as few test cases as possible, and improve the efficiency and results of the adequacy testing of intelligent systems. 
- **The platform layer** contains the development toolkit (PyQt5) and the development framework (TensorFlow framework) for implementing ISTA.

## Environment

##### This project is based on python 3.7 and tensorflow 2.6.0


## Video Demonstration

[Link to Video Demonstration](https://www.youtube.com/watch?v=6CkzMJ0ghq8)

