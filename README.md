# AI in Education Hackathon 2024
This project aims to analyze the fairness in student performance dataset [dataset](https://www.kaggle.com/datasets/rkiattisak/student-performance-in-mathematics?resource=download). With our project we will display a exploratory data analysis and build a machine learning model, which is able to predict scores on test data. By that we can verify, if the model learns a bias and then favors/disfavors certain student groups.

## Results
![Project idea:](https://github.com/arzx/bias-analysis-students/blob/main/project_plan.md) -> Here we define our scope and aims which we finished within the Hackathon.

![Exploratora Data Analysis (EDA):](https://github.com/arzx/bias-analysis-students/blob/main/notebooks/eda.ipynb) -> Here our inital data analysis can be found. From this we could decide if a bias could be a possibility and which machine learning model should be useful. 

![Random Forest and Linear Regression](https://github.com/arzx/bias-analysis-students/blob/main/notebooks/rf_lr_results.ipynb) -> Our first try to apply machine learning models to compare further experiments. This can be understood as a baseline. 

![Neural Network Results](https://github.com/arzx/bias-analysis-students/blob/main/notebooks/nn_results.ipynb) -> These are our final results, which show that there are obvious disparities in between the classes. But the neural network does not favor/disfavor any group and predicts the test data within their respective performance. 

![Story:](https://github.com/arzx/bias-analysis-students/blob/main/notebooks/story.ipynb) -> Here we summarized our approach and the answers to our key questions that we formulated before starting implementing the method.

## Installation
### (Recommended) Setup new clean environment
Use a conda package manager.

#### Conda
Subsequently run these commands, following the prompted runtime instructions:
```bash
conda create -n bias-analysis python=3.12.15
conda activate bias-analysis
pip install -r requirements.txt
```


## Used Hardware
| Compute/Evaluation Infrastructure    |                                      |
|:-------------------------------------|--------------------------------------|
| Device                               | MacBook Pro M3 Pro 14-Inch                  |
| CPU                                  | M3 Pro |
| GPU                                  | -                                    |
| TPU                                  | -                                    |
| RAM                                  | 18 GB RAM                       |
| OS                                   | Sonoma 14.5                        |
| Python Version                       | 3.12.15                      |

## Further Work
To further improve this project, the analysis can expand on other given classes and inspect their correlation. In order to process this visually, a dashboard would be effective.