# Disaster-Response-Pipeline
  Disaster Response Pipeline

This is a web application that may be used during disasters. The web application can take a message and determine which type of it and how can we take care of it.
The used data are real and are taken from Figure Eight. The project consists of three main parts. First, Data input is  performed using ETL pipeline, Second, Classifier is build using MLP pipeline. Finally, interacting with users using app application.



## The motivation for the project

The motivation for the project to help diffenet agencies in the case of disaster to interact with people who need help. 

##  Libraries used


| Library | Description |
| --- | --- |
| `numpy   : `   |  a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices |
| `pandas   : `    | a software library written for the Python programming language for data manipulation and analysis. |
| ` matplotlib.pyplo  : `      |   **a collection of command style functions that make matplotlib work like MATLABk** |
| ` sys  : `      |     |
| ` nltk   : `      |     |
| ` flask   : `      |     |
| `sqlalchemy    : `      |     |
| ` pickle   : `      |     |
| ` re   : `      |     |
| `  json  : `      |     |
| `plotly    : `      |     |


##  The files  in the repository are 

| FILE    | Description |
| ---  | --- |
| `messages     : `         |csv file that contains the messages  datasets  |
| `scategories     : `           |csv file that contains the  categories datasets  |
| `process_data.py  : `         |   **python file** data pipeline is the Extract, Transform, and Load process|
| `train_classifier.py  : `         |  **python file** machine learning portion, in which we split the data into a training set and a test set. |
| `run.py  : `         |  **python file** that consists of Flask App that will take care of other files|
| `README.md  : `                        |  the readme file that explain the usage of this repository|


## Summary of the results of the analysis

We conclude that about 70% of developers have a formal education of Bachelor’s or Master's degree. 
Developers with doctoral degrees earn the highest salaries.
70% of developers have a computer science or related field major in undergraduate. 
Non-computer science major in undergraduate earns more than computer science-related fields. 
Developers with a psychology undergraduate major got the highest salaries.
45 % of Developers’ parents have Bachelor’s or Master’s degree. 
The developers whose parents hold doctoral degrees earn the highest salaries.

### Installing


No installation is needed, since this is a Jupyter notebook that contains both
the code and its output. Also, dataset used is provided.

It recommended to use Anaconda distribution to install both Python 
and the notebook application. 


## Usage

Use Anaconda distribution to to run the Python Jupyter notebook.


## Built With

* Anaconda distribution using Jupyter notebook 


## Authors

* **Mohammed Lafi** - *  * - [Mohammed Lafi](https://github.com/mohammedlafi)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

I would like to thank ALL Udicity team members 

