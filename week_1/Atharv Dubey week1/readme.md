# Week 1 TASKS

Welcome to the WIN-FACTOR! This Week 1 assignment will give you hands-on practice with:

* **Data Visualization** using Matplotlib and Seaborn (histograms, boxplots, scatterplots, pairplots).
* **Skewness & Normalization** (understanding distribution skewness and applying log, MinMaxScaler, StandardScaler transformations).
* **Clustering** (simple K-Means clustering on a sample dataset, e.g. customer segments by income and spending).

Take your time, play around with the code, and ask questions if you get stuck. **Please submit your work by *May 20***.


## Part 1: Visualization with Matplotlib and Seaborn

Use Python plotting libraries to explore the data. Make sure to label axes and titles on your plots.

* **Histograms:** Plot the distribution of key numerical columns.
* **Boxplots:** Use box plots to see quartiles and outliers.[How to read boxplots](https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/data-presentation/box-and-whisker-plots.html)
* **Scatter plots:** Plot relationships between two variables.
* **Pairplot:** Use Seaborn’s pairplot for multivariate EDA.
For more details, see the [Matplotlib tutorial](https://matplotlib.org/stable/tutorials/introductory/pyplot.html) and Seaborn documentation.
- [different types of plots](https://www.comet.com/site/blog/different-plots-used-in-exploratory-data-analysis-eda/)
- [code for different plots](https://medium.com/artificial-intelligence-101/exploratory-data-analysis-eda-with-matplotlib-and-seaborn-a-practical-guide-9284672534df)
- [notebook for seaborn eda](https://www.kaggle.com/code/abdokamr/eda-by-seaborn-feature-engineering-for-beginners)


## Part 2: Skewness and Normalization
* **Check skewness:** Look at your histograms. A right-skewed distribution has a long tail to the right, while a left-skewed distribution has a tail to the left. Normal distributions have (approximately) zero skewness.
* **Apply transforms:** If a feature is skewed, try transforming it:
  * *Log transform*
  * *StandardScaler*
  * *MinMaxScaler*
* **Compare distributions:** After scaling or transforming, re-plot the histograms to see the effect. Ideally, the transformed data should appear more normally distributed.
* **Document results:** Note how the mean and standard deviation changed after scaling, and whether the skewness improved.
- [skewness and its types](https://yashowardhanshinde.medium.com/what-is-skewness-in-data-how-to-fix-skewed-data-in-python-a792e98c0fa6)
- [different types of transformations](https://www.linkedin.com/pulse/how-remove-skewness-data-learn-6-powerful-siddharth-sahasrabudhe-xla3f?trk=public_post)
- [notebook for removing skewness](https://www.kaggle.com/code/swagabyss/skew-handling-techniques#Methods-for-handling-the--Right-Skewed-data)
- [reddit discussion on why normalizaton](https://www.reddit.com/r/datascience/comments/v4u6e4/why_should_we_normalize_our_data_are_there_any/)
- [normalization](https://medium.com/@weidagang/demystifying-machine-learning-normalization-0cdb8b281234)
## Part 3: Clustering with K-Means

Perform a simple customer segmentation using K-Means clustering:
1. **Select features:** For example, cluster on `Annual Income (k$)` and `Spending Score (1-100)`.
2. **Run K-Means:**
   K-Means will divide the data into 3 clusters based on feature similarity.
3. **Plot clusters:**
4. **Interpretation:** Examine the cluster centers (`kmeans.cluster_centers_`) or look at the average income/score in each cluster. Can you describe each segment (e.g., high-income low-spenders, medium-income medium-spenders, etc.)?
5. **Optional:** Try a different number of clusters (K) or use the elbow method to choose K.

K-Means is a basic unsupervised algorithm that groups unlabeled data into clusters. It's widely used for tasks like customer segmentation.
- [what is clustering](https://developers.google.com/machine-learning/clustering/overview)
- [types of clustering](https://developers.google.com/machine-learning/clustering/clustering-algorithms)
- [k means clustering](https://medium.com/@dishantkharkar9/k-means-clustering-algorithm-ce4fbcac8fb0), [another article](https://medium.com/analytics-vidhya/everything-you-need-to-know-about-k-means-clustering-88ad4058cce0)
- [notebook for k means clustering](https://www.kaggle.com/code/prashant111/k-means-clustering-with-python#12.-K-Means-model-with-two-clusters-)
- [k means clustering by statquest](https://youtu.be/4b5d3muPQmA?si=9YSNWobmqy4djpYP)
## Deliverables

* **Jupyter Notebook:** `week0_exploration.ipynb` containing your code, outputs, and explanations (in Markdown).
* **Data (if changed):** If you generated a new CSV (e.g., with transformed or scaled columns), include it.
* **Plots:** Ensure all generated plots (histograms, boxplots, scatterplots, pairplot, cluster plot) are visible in the notebook.
* **Summary:** A brief summary (a few sentences) of your observations (e.g., which features were skewed, effect of transformations, and what the clusters represent).

Submit these materials by **May 18**.

## Stretch Goals (Optional)

* Try other skew-reduction transforms (e.g., square root, Box–Cox) on skewed features.
* Cluster using more features and visualize with a 3D plot or additional pairplot.
* Apply these steps to a different dataset (e.g., Iris or Wine) for practice.
* Implement an elbow plot to find the optimal number of clusters.
* Customize your plots (colors, styles, annotations) for a polished look.

Enjoy this introductory exploration! We’re excited to see your visualizations and insights. Good luck and have fun with the data!

**References:** Basic plotting and EDA concepts are covered in Matplotlib/Seaborn tutorials.