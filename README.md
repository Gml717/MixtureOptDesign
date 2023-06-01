# MixtureOptDesign - Installation Guide

This guide will walk you through the steps to install the MixtureOptDesign package on your system. Please follow the instructions below based on your operating system.

## Windows

### Prerequisites


1. Python 3 can be downloaded at [python.org] (https://www.python.org).

2. C++ Builder: You need install C++ Builder to build the Cython components of the package. Download [visual studio code](https://visualstudio.microsoft.com/) and choose the option desktop development with C++ and click on download all the packages.

### Installation Steps

1. Download: Download the zip file.
2. Extract: Extract the contents of the zip file to a directory.

#### Using Virtual Environment

1. Create Virtual Environment: Open a Command Prompt window and navigate to the extracted directory. Run following command to create a virtual environment:

   ```
   python -m venv env
   ```

2. Activate the Virtual Environment: Activate the virtual environment by running the following command:

   ```
   env\Scripts\activate
   ```

3. Install Packages: With the virtual environment activated, run the following command to install requirements for MixtureOptDesign:

   ```
   pip install -r requirements.txt
   ```
4. Install MixtureOptDesign: With the virtual environment activated, run the following command to install MixtureOptDesign:

   ```
   pip install -e .
   ```
5. Verify Installation: Once the installation is finished, you may check that it was successful by importing the package into a Python script or Python's interpreter.



## Mac

### Prerequisites

1. Python 3 can be downloaded at [python.org] (https://www.python.org).

2. C++ Builder: You need install C++ Builder to build the Cython components of the package. Download visual studio code [visual studio code](https://visualstudio.microsoft.com/) and choose the option desktop development with C++ and click on download all the packages.

### Installation Steps

1. Download: Download the zip file.
2. Extract: Extract the contents of the zip file to a directory.

#### Using Virtual Environment

1. Create Virtual Environment: Open a Command Prompt window and navigate to the extracted directory. Run following command to create a virtual environment:

   ```
   python3 -m venv env
   ```

2. Activate the Virtual Environment: Activate the virtual environment by running the following command:

   ```
   source package_env/bin/activate
   ```

3. Install Packages: With the virtual environment activated, run the following command to install requirements for MixtureOptDesign:
   ```
   pip install -r requirements.txt
   ```
4. Install MixtureOptDesign: With the virtual environment activated, run the following command to install MixtureOptDesign:

   ```
   pip install -e .
   ```
5. Verify Installation: Once the installation is finished, you may check that it was successful by importing the package into a Python script or Python's interpreter.
