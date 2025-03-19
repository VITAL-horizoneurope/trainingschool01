# VITAL Training School 01
March 18-21 2025, Delft Netherlands

Coordinator/organizer: Mathias Peirlinck  
Lecturers: Mathias Peirlinck, Peter Hunter, Finbar Argus, Beatrice Ghitti, Nikolaos Stergiopulos, Lydia Aslanidou, Wouter Huberts, Shauna O'Donovan, Alberto Zingaro, Gonzalo Maso Talou, Dimitrios Lialios, Rogier Krijnen

## Installation instructions
### OpenCOR/Circulatory Autogen (days 1, 3, and 4)
For Instructions on installation of OpenCOR and the python setup needed to run Circulatory Autogen please go to ([getting-started](https://finbarargus.github.io/circulatory_autogen/getting-started/))

### 0D calibration and deep learning (day 2)
Please download the Pulse Wave Data Base (PWDB) mat file, available at the end of this page https://zenodo.org/records/3275625, and place it in the `data` subfolder of the tutorial.
You can prepare the python environment using the yaml file located in ([trainingschool01/20250319-day02/0Dcalibration_deeplearning_tutorial](https://github.com/VITAL-horizoneurope/trainingschool01/tree/main/20250320-day02/0Dcalibration_deeplearning_tutorial)).

### Bayesian inference workshop (day 3)
We prepared a Python environment for you to run our Bayesian Inference Jupyter notebooks,  
located in ([trainingschool01/20250320-day03/peirlincklab_BI](https://github.com/VITAL-horizoneurope/trainingschool01/tree/main/20250320-day03/peirlincklab-BI)).  
Please install this environment beforehand.

### Surrogate Modelling (day 4)
The tutorial session will require to:
- Have basic knowledge of Python coding;
- Create a Python virtual environment with TensorFlow (2.8 or higher) installed to execute the tutorial codes;
- Have enough knowledge of TensorFlow to modify network inputs and to switch network inputs with variables ([suggested TF documentation](https://www.tensorflow.org/guide/variable));
- Have enough knowledge of TensorFlow to understand neural network training loops ([suggested TF tutorial](https://www.tensorflow.org/tutorials/quickstart/advanced));
- Download the tutorial codes host at [Animus Lab repositories](https://github.com/ABI-Animus-Laboratory/AI_surrogate_tutorial/tree/main).

Ideally, but not strongly required, we encourage you to:
- use a UNIX-like SO (this will enable us to give you better support during the tutorial);
- use an IDE such as PyCharm, VS Code or Eclipse pyDev (this will facilitate the coding and execution exercises);
- bring a computer with an NVIDIA® GPU card with CUDA® architectures 3.5, 5.0, 6.0, 7.0, 7.5, 8.0 and higher (this will speed up the execution of the scripts).


### Alya workshop 

The alya workshop will require:

- a distribution of `conda`.
- a distribution of [ParaView](https://www.paraview.org/download/).
