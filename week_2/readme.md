# Week 2 TASKS

## Week 2 Assignment
A change this week! You have been assigned groups to work with. Find your groups in [this sheet](https://docs.google.com/spreadsheets/d/1u9QVW0PpTno4jXh7bDajUXJOC_3LUGvOPvMpTGdEJH0/edit?usp=sharing)

This Week 2 assignment will give you hands-on practice with:

### Supervised Learning and Regression Models
- Understanding the difference between **supervised** and **unsupervised** learning  
- Implementing **linear** and **polynomial regression** models  
- **Evaluating model performance** using appropriate metrics (e.g., R² score, MAE, MSE)

### Neural Networks (Basics)
- Getting started with **neural network fundamentals**  
- Understanding the concept and mechanism of **backpropagation**  
- Exploring the **mathematics behind neural networks**, including forward pass and gradient computation

---

Take your time to experiment with the models and concepts. Use visualizations or code-based intuition to reinforce your understanding.

**Please submit your work by *May 29*.**

## Resources
- [What is supervised learning?](https://www.ibm.com/think/topics/supervised-learning)
- [What are regression models?](https://www.iotaacademy.in/post/how-regression-models-are-used-in-supervised-learning)
- [linear and polynomial regression](https://medium.com/analytics-vidhya/everything-you-need-to-know-about-linear-regression-750a69a0ea50)||[another good article](https://www.analyticsvidhya.com/blog/2021/10/everything-you-need-to-know-about-linear-regression/)||[another one](https://medium.com/analytics-vidhya/understanding-the-linear-regression-808c1f6941c0)
- [linear regression by statquest](https://youtu.be/7ArmBVF2dCs?si=vu4mJk738x1AfQps)
- [vdeo resource on polynomial regression](https://youtu.be/BNWLf3cKdbQ?si=P70nrkrrrW6QA3tp)
- [3B1B Neural Networks Playlist (Highly recommended)](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=TnAAsjDxla7tiKzc), then go through [this](https://medium.com/deep-learning-demystified/introduction-to-neural-networks-part-1-e13f132c6d7e) and [this](https://medium.com/deep-learning-demystified/introduction-to-neural-networks-part-2-c261a99f4138)
- [gradient descent](https://towardsdatascience.com/gradient-descent-unraveled-3274c895d12d-2/)
- [types of optimization in gradient descent](https://arxiv.org/pdf/1609.04747), only for the interested peeps :)
- [activation functions and their types](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)
- [how to choose an activation function](https://www.turing.com/kb/how-to-choose-an-activation-function-for-deep-learning)
- [types of loss fucntions](https://www.ibm.com/think/topics/loss-function)
- [why mse though?](https://blog.dailydoseofds.com/p/why-mean-squared-error-mse), only for the interested peeps :)
## Task 1: Familiarize with the Dataset
- You have been provided with a dataset - `mw_pw_profiles.csv` - matchwise playerwise dataset
- This is a large dataset with multiple features, familiarize yourself with the columns and their meanings.
- Think about how you could use various columns to develop your own features. 
- Play around in a notebook with **pandas** to explore the dataset.

## Task 2: Implement Linear Regression
- Implement a simple linear regression model using Python with the provided dataset (this is to be done from scratch without using any libraries like scikit-learn, pytorch, etc).
- Implement polynomial regression to capture non-linear relationships.

## Task 3: Implement Neural Networks on the Dataset
- Implement a neural network model using Python with the provided dataset (any architecture is fine)

## Deliverables
- **Jupyter Notebook:** `week2_exploration.ipynb` containing your explorations with the dataset.
- Linear Regression Model implemented from scratch: `linear_regression.py`
- Polynomial Regression Model implemented: `polynomial_regression.py`
- Neural Network Model implemented: `neural_network.py`
