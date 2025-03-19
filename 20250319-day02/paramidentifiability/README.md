<img src="figures/TUe.png" width="280" height="70"> <img src="figures/EAISI.png" width="200" height="60">

# Parameter Indentifiability: Personalising the Eindhoven Diabetes Education Simulator
This is the repository for the Tutorial on parameter uniqueness/identifiability as part of the VITAL Training School 1, Delft University of Technology on March 19th 2025. 

## About the Tutorial
Computational models are a valuable tool to study dynamic interactions and the evolution of systems behavior. Our hands-on and interactive workshop will demonstrate how personalized models can be more rapidly generated in Julia using various SciML packages combined with custom implementations. We will cover the implementation of ODE models in Julia, parameter estimation and model selection strategies including parameter sensitivity and identifiability analysis.

Computational models offer a valuable tool for understanding the dynamic interactions between different biological entities, especially in biomedical applications. Personalizing these models with data can shed light on interindividual variation and project future health risks. However, for model parameters to be meaningful and useful for interpretation it is important that they can be accuractely identifed from available data. Our hands-on and interactive workshop will demonstrate how personalized models can be more rapidly generated in Julia. We will be mainly using DifferentialEquations.jl combined with Optimization.jl. We will use an in-house model of the glucose-insulin system (Eindhoven Diabetes Education Simulator) we will breifly cover the implementation and resolving of ODE systems in Julia. We will provide a guide on performing identifiability analysis using inbuilt Julia packages such as StructuralIdentifiabilty.jl and a implementation of Profile Likelhood Analysis [2] with a variable step. For Matlab users and implementation of the model and PLA algorithm can also be provided in Matlab. Short presentation will be used to provide necessary background and theory and all methods will be implemented in a Jupyter notebook to facilitate independent learning.


## Organizers
Shauna O'Donovan |  Max de Rooij | 
|:---:|:---:|
<img src="figures/O_Donovan_Shauna_BME_PO_VH_1529_UD.jpg.webp" width=90 height=120>  | <img src="figures/de_Rooij_Max_BME_PROM_PO_AS_7631.jpg.webp" width=90 height=120> | 

## Contents
During the workshop, we will address the following elements of dynamic modelling in (systems) biology:

Implementation and simulation of (biological) dynamic models using DifferentialEquations.jl.  
Parameter estimation in Julia.  
Parameter identifiability analysis with profile likelihood. 

## Program
The workshop starts with a short introductory presentation, outlining the releavnce of the gluocse-insulin system in maintaining metabolic resilence, the Eindhoven Diabetes Education Simulator (EDES), a computational model of the glucose-insulin system that has been shown to capture relevant features of metabolic resilence from measures time series of glcuose and insulin. The workshop will then explain the principles of parameter identifiablity and provide hands on experience using in-built Julia packages for determining parameter identifiability and an inplementation of the Profile Liklihood Analysis methods proposed by Raue et al [2].

# Julia Installation and Environment Setup
Here you will find some instructions on how to make sure the code from this repository can run on your computer. The workshop is built to fully work with Julia version 1.9. In this instruction, we will assume you are installing Julia 1.9, but any other version is installed in a similar way. 


### Installing Julia using the Julia version manager (juliaup) (recommended)
The first step is to install the Julia version manager (`juliaup`), which enables you to have multiple Julia versions installed on your machine. This step is not mandatory for a properly working version of Julia, but it is useful if you are planning to use Julia for your own projects in the future.

**Windows**:
>  [!NOTE]
>  The `winget` command may not be available on your machine! In that case, you can download Julia from the Microsoft Store. If your organization prevents use of the Microsoft Store, you can follow the download instructions from https://julialang.org when clicking `Download`

Open your favorite terminal, or press `Win`+`R`, type `cmd` in the text bar and press 'run'.
```
winget install julia -s msstore
```

**Linux & MacOS**:

From the command line execute
```
curl -fsSL https://install.julialang.org | sh
```
---

Afterwards, you can restart the terminal environment. You can install a specific version of Julia using
```
juliaup add <version>
```

You can list the versions available to you using
```
juliaup list
```

We recommend to start with installing the latest stable version, dubbed `release` by running
```
juliaup add release
```

If this is your only Julia version, juliaup will automatically label this as the default version. 

---

### Setting up VSCode
Julia was found to work the best using the VSCode IDE, which you can download [here](https://code.visualstudio.com/download). After downloading and installing VSCode, navigate to the VSCode marketplace, which has the following icon:

<img width="54" alt="VSCode Marketplace" src="https://github.com/Computational-Biology-TUe/Julia-sysbio-workshop/assets/54850292/4f1ce454-ce4e-47c4-8a92-bfc636e48140">

Type `Julia` in the search bar and install the Julia Language Support extension. 

---

### Setting up the workshop environment
After installing Julia, you can either fork this repository to your own Github account and download the code from there, or download the code directly from this repository into a Zip-file by selecting `Code` -> `Download ZIP`. 

Put the workshop code in a nice folder and open this folder from VSCode.

Now there are two ways to initialize the environment. 

#### 1. Use the first code block in the 01-Getting-Started file
Uncomment the first code block in the `01-Getting-Started.ipynb` file to automatically set up the environment.

#### 2. Use the VSCode terminal

In case a terminal is not open yet, click `Terminal`->`New Terminal` from the top bar and execute
```
julia
```
To start the Julia REPL.

Then, press `]` to change from Julia to the package manager, which is indicated by `>julia` changing to `>pkg`. Then, execute:
```
activate .
```
>  **Warning**
> Make sure to remember the dot `.` at the end of this command.

To activate the workshop environment. Then execute (also from the Pkg terminal)
```
instantiate
```

To install all the required packages. You should now be able to run all the notebooks in this workshop environment.

---

### Troubleshooting

#### Error opening notebooks in Windows VS Code
In Windows, we noticed that notebooks may sometimes error when trying to open them in Visual Studio Code. This has to do with the path length limit in the Windows filesystem. The easiest fix is to place the repository folder in a directory that is higher up, so the paths will not be too long. 

## References 
[1]Maas AH, Rozendaal YJ, van Pul C, Hilbers PA, Cottaar WJ, Haak HR, van Riel NA. A physiology-based model describing heterogeneity in glucose metabolism: the core of the Eindhoven Diabetes Education Simulator (E-DES). J Diabetes Sci Technol. 2015 Mar;9(2):282-92. doi: 10.1177/1932296814562607.  
[2] Raue A, Kreutz C, Maiwald T, Bachmann J, Schilling M, Klingmüller U, Timmer J. Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood. Bioinformatics. 2009 Aug 1;25(15):1923-9. doi: 10.1093/bioinformatics/btp358. 
