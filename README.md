# Bank telematketing ML

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


This repository contains a machine learning project to analyze and predict the outcomes of a bank telemarketing campaign. The goal is to identify which clients are most likely to subscribe to a term deposit. This project is part of the **models II** course and is presented by the students:

- Silvio José Otero Guzman.
- Gaia Ramirez Hincapié.
- Sara Galván Ortega.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Implemented Models](#implemented-models)
- [Contributing](#contributing)
- [License](#license)

## Requirements

Before installing and running the project, make sure you have:

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning the repository)

## Installation

1. **Clone the repository** (optional):

```bash
git clone https://github.com/galvanic90/bank_telemarketing_ML.git
cd bank_telemarketing_ML 

```

2. **Create a virtual environment**

This step is very recommended becouse this way, we avoid possible errors from python libraries previously installed.

```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate  # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Project Structure

```bash
bank_telemarketing_ML/
├── data/                   # Original   dataset             
├── src/                    # Python source code
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── model_evaluation.py
├── informe_final/
│   └── entrega2_proyect_modelos_2.pdf
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## Implemented Models

The project includes implementations of:

* Logistic Regression
* Random Forest
* SVM
* Neural Networks