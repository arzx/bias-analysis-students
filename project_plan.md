# Bias Analysis towards Fair AI by utilizing student performance dataset
We want to showcase the correlation between student performance based on their ethnicity/race background and their chances of respective milestones. Additionally we will analyze the influence of the parental educational background on the students performance. By this analysis, we can showcase the inequality of chances.

The challenge will be to select a suitable machine learning model that will be able to detect the underlying pattern in the data without much adjustment. Another challenge will be to clean the data and divide it into test/validate/train. One possible solution could be ensamble learning instead of going for one specific ML-model. The first step to solve this would be whether we will formulate this as a classification problem or as a regression. To reduce pitfalls, we use pre-existing ML-models and decide a model that does not overfit on a clean or unprocessed dataset.

### Target group
By our analysis, we want to provide insight to anyone interested in educational analysis and researchers/students who are eager to work towards fair AI. Most possibly the interest grew because they are teaching and want to create fair chances and or they are affected. By successfully understanding the data, the target group can adjust their methods (e.g., assigning groups in a more diverse fashion). The goal is worth investigating because there might be improvement in educational environments where every background can prosper and adding their knowledge and skill.

Questions that can be drawn from this task formulation:
1. Are there disparities in performance between each inspected group?
2. Is there any reason of doubt that the resulting correlations could be assumed of unfair chances?
3. How can educators use these findings to create fairer educational opportunities and reduce biases?

### Goals and Objectives
Our goal is to showcase the inequality of chances for different groups. By showing the performance of existing data entries on different classes and prediction of new introductions. The potential impact of that is to motivate educators to investigate their own situation and strive for improvement if our results align with the reality.

We will verify our solution by showing that our assumption based on our initial analysis is met or not met by our prediction. To evaluate our results we can monitor key metrics like accuracy, precision and use K-Fold Cross-Validation. Further evaluation metrics like ROC-AUC and MSE/MAE are also possible, depending on our final architecture. Furthermore we can use bootstrapping to estimate the final distribution and repeatedly re-sample data into our model with a final aggregation. Statistical tests can verify our results (e.g. t-test, etc.).

### Requirements

1. A reasonable division of the dataset (train, validate, test)
2. if necessary: do pre-processing
3. Using classification or regression models
4. Validate that there is no over-/underfitting by monitoring the loss
5. Provide interpretable plots
6. Ensure reproducibility (adding seeds, document the used software and computer hardware)

### Methodology
1. For dividing the dataset we will use K-Fold Cross-Validation with a 70-20-10 split with K = 10
2. Possibilities: add missing labels, synthesize missing data or ensure equal balancing
3. We will experiment with different Machine Learning models.
Possible model decisions:
    - Random Forests
    - Decision Trees
    - Linear/Logistic Regression
    - K-Means Clustering
    - Neural Network
4. We will monitor the loss, accuracy and precision. By that we can ensure that there is no over/-underfitting
5. We will create several plots with matplotlib, seaborn. Depending on what sub-problem we want to show, the plots can be very different (e.g. 2D, 3D, histograms, graphs, scatterplots, etc.)

We will propose our solution with a set of plots (visual representations of our statistical evaluation) which will be our foundation to showcase our project statement.
We found the data on: 

- Kaggle - Student performance prediction
- https://www.kaggle.com/datasets/rkiattisak/student-performance-in-mathematics


### Timeline
First we will create a initial analysis to get familiar with the data. By that we can decide if pre-processing is necessary.

Afterwards we can apply K-Fold Cross-Validation and start to do a first approach towards apply ML models. The result metrics (loss, accuracy, precision) will be the indicator whether we need to adjust the model.

Once this is done, we can use the metrics to plot them and finalize our project. The most important thing here is to generate insights that go beyond trivial statements. For example to visualize which group has a significant different performance beyond randomness and to perform successful predictions with a confidence interval. This will align with our overall goal to investigate possible disparities of performances between selected groups.

Milestones:
1. Data preprocessing step: initial analysis of the data, identification of potential issues and readjustment: potential time -> 30 min - 2 hours
2. Implement different ML models and compare the outcomes to decide which model serves the solution: 3hours - 5hours
3. Inspect the found metrics and process it into plots: 2 - 3hours

### Evaluation Plan  
Milestone 1: Data preprocessing:
- We need to make sure that the data is not imbalanced and contains enough samples for each class. Reduce/Add duplicates if they are distrorting the final solution (over-/underfitting).

Milestone 2: Implementing ML models:
- We need to monitor the loss, accuracy and precision to draw conclusions about the performance. These can be storeed in a JSON or an array.
- We need to make to make sure that we don't overengineer the final ML-model so we save on computational resources. This can be evaluated by testing it on a smaller network and compare the final result of the metrics.

Milestone 3: Data presentation:
- Adjust the plot labels to make them interpretable for the target group by showing the plot to someone from the target group

### Risks
The main risk of this project that there is no non-trivial insightt in the data and we can't draw any conclusion out of it. Although that is not realistic, it is still possible. The more possible risk is that we wont be able to ensure enough computational resources to solve  the task in the given timeframe. If that happens, we need to break down to a simpler ML model and work with a potentially lower accuracy.
