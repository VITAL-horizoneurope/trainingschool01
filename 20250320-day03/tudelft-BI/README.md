# README: VITAL training school on Bayesian inference
This readme will guide you through installing the environment file for these jupyter notebooks.
The environment file itself also contains this package so there is no need to install this beforehand.

The environment file has been created using Conda which means that installing it will be easiest if you have Conda installed.
After git cloning this repository open a terminal in the folder and execute the following command:
```
conda env create -f ./environment.yml 
```
This may take a while because we use Pytorch, which usually takes a while to isntall (should not take longer than 5 minutes).
When the installation process has finished you will need to activate the environment.
In VScode this would require you to open the folder containing the repository and selecting a jupyter kernel containing the newly created conda environment.
The kernel prompt is usually in the top right corner of your editor.

