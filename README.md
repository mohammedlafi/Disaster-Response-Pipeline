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

We Extrcted data from two datafiles messages and categories, we transofromr them then we save them as databas.
Then, we read the database into dataframes, build machine learning model, train the model and test it.
We used flask app to goup all together.

### Installing


unning the Web App from the Project Workspace IDE
When working in the Project Workspace IDE, here is how to see your Flask app.

Open a new terminal window. You should already be in the workspace folder, but if not, then use terminal commands to navigate inside the folder with the run.py file.

Type in the command line:

python run.py
Your web app should now be running if there were no errors.

Now, open another Terminal Window.

Type

env|grep WORK


## Usage

or run 
https://view6914b2f4-3001.udacity-student-workspaces.com

![screen1](https://user-images.githubusercontent.com/19904555/70387929-d44edd00-19b3-11ea-8e43-551942fe045e.PNG)
![screen2](https://user-images.githubusercontent.com/19904555/70387931-d44edd00-19b3-11ea-8089-4ce6474c1469.PNG)
![screen3](https://user-images.githubusercontent.com/19904555/70387932-d4e77380-19b3-11ea-981d-15d8db7e4ce8.PNG)



## Built With

* Anaconda distribution using Jupyter notebook 
* Flask


## Authors

* **Mohammed Lafi** - *  * - [Mohammed Lafi](https://github.com/mohammedlafi)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

I would like to thank ALL Udicity team members 

