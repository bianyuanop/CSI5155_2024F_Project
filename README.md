# CSI5155_2024F_Project
Haonan Chen (hchen312@uottawa.ca)
Eric Sun (ysun338@uottawa.ca)
### **Summary of Best Hyperparameter Settings for Gradient Boosting and Semi-Supervised Learning Models**

Below is a summary of the best hyperparameter settings for the five models utilized in our experiments on the Magic Mushrooms dataset. The table distinguishes between the supervised Gradient Boosting baseline and the four semi-supervised learning (SSL) methods, highlighting both the optimized and default hyperparameters used.

| **Model**                         | **Base Estimator**                | **Key Hyperparameters**                                                                                                                                                                                                                                                                                                                                                     |
|-----------------------------------|------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Gradient Boosting (Baseline)**  | `GradientBoostingClassifier`      | - **learning_rate**: `0.01` <br> - **max_depth**: `3` <br> - **max_features**: `None` <br> - **min_samples_leaf**: `5` <br> - **min_samples_split**: `20` <br> - **n_estimators**: `200` <br> - **subsample**: `1.0`                                                                                                                                                                |
| **Self-Training**                 | `LogisticRegression`               | - **Threshold**: `0.75` <br> - **Criterion**: `'threshold'` <br> - **k_best**: `10` <br> - **max_iter**: `10` <br> - **verbose**: `False` <br> *Default hyperparameters as provided by scikit-learn’s `SelfTrainingClassifier`*                                                                                                                                                |
| **Co-Training**                   | `LogisticRegression` & `DecisionTreeClassifier` | - **Number of Iterations (n_iter)**: `10` <br> - **Number of Positive Samples per Iteration (p)**: `5` <br> - **Number of Negative Samples per Iteration (n)**: `5` <br> - **Feature Views**: <br> &nbsp;&nbsp;&nbsp;- *Estimator 1*: First 6 features (`0-5`) <br> &nbsp;&nbsp;&nbsp;- *Estimator 2*: Last 6 features (`6-11`) <br> *Custom hyperparameters set for the `CoTrainingClassifier`* |
| **Semi-Supervised Ensemble**      | `DecisionTreeClassifier` (10 voters) | - **Number of Estimators (n_estimators)**: `10` <br> - **Voting Mechanism**: `'soft'` <br> *Default hyperparameters as provided by the custom `SemiSupervisedEnsembleClassifier` implementation*                                                                                                                                                |
| **Pretrained Autoencoder**        | `SVC`                              | - **Autoencoder Hidden Layer Sizes**: `(5,)` <br> - **Autoencoder Activation**: `'relu'` <br> - **Autoencoder Solver**: `'adam'` <br> - **Autoencoder Learning Rate Init**: `0.001` <br> - **Autoencoder Max Iterations**: `200` <br> - **SVM Probability**: `True` <br> - **SVM Random State**: `42` <br> *Default hyperparameters as provided by scikit-learn’s `SVC` and custom `SemiSupervisedAutoencoderClassifier`* |

#### **Details and Considerations:**

1. **Gradient Boosting (Baseline):**
   - **Optimization:** The Gradient Boosting model underwent hyperparameter tuning to identify the optimal configuration that maximizes performance on the labeled dataset.
   - **Selected Parameters:** A low learning rate of `0.01` combined with a moderate `max_depth` of `3` and a substantial number of estimators (`200`) ensures a balance between bias and variance, enhancing the model's generalization capabilities.

2. **Self-Training:**
   - **Default Settings:** Leveraging scikit-learn’s `SelfTrainingClassifier`, this method uses a `LogisticRegression` base estimator with a threshold of `0.75` to iteratively label the most confident unlabeled instances.
   - **Impact of Defaults:** The default parameters facilitate a controlled expansion of the labeled dataset, promoting stability in learning while incorporating valuable information from unlabeled data.

3. **Co-Training:**
   - **Custom Configuration:** Utilizing two distinct classifiers—`LogisticRegression` and `DecisionTreeClassifier`—this method splits the feature set into two views (first 6 and last 6 features) to enable complementary learning.
   - **Parameter Choices:** Setting `n_iter=10`, `p=5`, and `n=5` allows the algorithm to iteratively incorporate a fixed number of positive and negative samples from the unlabeled data, fostering robust co-learning between the classifiers.

4. **Semi-Supervised Ensemble:**
   - **Ensemble Structure:** Comprising 10 `DecisionTreeClassifier` voters with a `'soft'` voting mechanism, this ensemble aggregates probabilistic predictions to enhance classification accuracy.
   - **Default Hyperparameters:** The ensemble leverages scikit-learn’s default settings within the custom `SemiSupervisedEnsembleClassifier`, ensuring a straightforward and effective aggregation of multiple classifiers’ strengths.

5. **Pretrained Autoencoder:**
   - **Integrated Learning:** This method first trains an autoencoder (`MLPRegressor` with hidden layer size `(5,)` and `'relu'` activation) to learn compressed representations of the data, subsequently applying a self-training `SVC` classifier with probability estimates enabled.
   - **Default Parameters:** The autoencoder’s default configuration enables effective feature extraction, while the `SVC`’s probability setting (`probability=True`) facilitates accurate AUC computations during evaluation.

#### **Conclusion:**

The selection and tuning of hyperparameters play a crucial role in the performance of both supervised and semi-supervised models. While the Gradient Boosting baseline was meticulously tuned to achieve optimal performance, the semi-supervised models primarily relied on default settings provided by scikit-learn, supplemented with custom configurations where necessary. This approach underscores the balance between leveraging established defaults for robustness and introducing tailored parameters to enhance specific learning paradigms within SSL frameworks.

By comparing these models, it becomes evident that semi-supervised methods can effectively harness unlabeled data to approach or even surpass supervised baselines, especially when the models are appropriately configured to exploit the inherent structure of the data.
