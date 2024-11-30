import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
from sklearn.utils.validation import check_X_y, check_array

#co-training
class CoTrainingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator1, estimator2, n_iter=10, p=1, n=1, 
                 view1_features=None, view2_features=None):
        self.estimator1 = clone(estimator1)
        self.estimator2 = clone(estimator2)
        self.n_iter = n_iter
        self.p = p  # Number of positive samples to add each iteration
        self.n = n  # Number of negative samples to add each iteration
        self.view1_features = view1_features
        self.view2_features = view2_features

    def fit(self, X_labeled, y_labeled, X_unlabeled):
        # Initialize labeled data for each estimator
        L1_X = X_labeled.copy()
        L1_y = y_labeled.copy()
        L2_X = X_labeled.copy()
        L2_y = y_labeled.copy()
        U_X = X_unlabeled.copy()

        # Initialize feature views
        if self.view1_features is not None:
            L1_X_view = L1_X[:, self.view1_features]
        else:
            L1_X_view = L1_X

        if self.view2_features is not None:
            L2_X_view = L2_X[:, self.view2_features]
        else:
            L2_X_view = L2_X

        U_index = np.arange(len(U_X))

        for i in range(self.n_iter):
            # Extract features for estimator1
            if self.view1_features is not None:
                U_X_view1 = U_X[:, self.view1_features]
            else:
                U_X_view1 = U_X
            
            # Train estimator1
            self.estimator1.fit(L1_X_view, L1_y)
            # Predict on U
            probs1 = self.estimator1.predict_proba(U_X_view1)
            # Find most confident positive and negative predictions
            pos_confidence1 = probs1[:, 1]
            neg_confidence1 = probs1[:, 0]
            pos_indices = np.argsort(-pos_confidence1)
            neg_indices = np.argsort(-neg_confidence1)
            # Select top p positive and n negative samples
            pos_samples1 = min(self.p, len(pos_indices))
            neg_samples1 = min(self.n, len(neg_indices))
            pos_idx1 = pos_indices[:pos_samples1]
            neg_idx1 = neg_indices[:neg_samples1]
            idx1 = np.concatenate([pos_idx1, neg_idx1])
            # Get corresponding samples and labels
            new_samples1 = U_X[idx1]
            new_labels1 = np.array([1]*pos_samples1 + [0]*neg_samples1)
            # Remove selected samples from U_X
            mask = np.ones(len(U_X), dtype=bool)
            mask[idx1] = False
            U_X = U_X[mask]
            U_index = U_index[mask]

            # Add new samples to L2
            L2_X = np.vstack([L2_X, new_samples1])
            if self.view2_features is not None:
                new_samples1_view = new_samples1[:, self.view2_features]
                L2_X_view = np.vstack([L2_X_view, new_samples1_view])
            else:
                L2_X_view = np.vstack([L2_X_view, new_samples1])
            L2_y = np.concatenate([L2_y, new_labels1])

            # If no more samples to label, break
            if len(U_X) == 0:
                break

            # Extract features for estimator2
            if self.view2_features is not None:
                U_X_view2 = U_X[:, self.view2_features]
            else:
                U_X_view2 = U_X

            # Train estimator2
            self.estimator2.fit(L2_X_view, L2_y)
            # Predict on U
            probs2 = self.estimator2.predict_proba(U_X_view2)
            # Find most confident positive and negative predictions
            pos_confidence2 = probs2[:, 1]
            neg_confidence2 = probs2[:, 0]
            pos_indices2 = np.argsort(-pos_confidence2)
            neg_indices2 = np.argsort(-neg_confidence2)
            pos_samples2 = min(self.p, len(pos_indices2))
            neg_samples2 = min(self.n, len(neg_indices2))
            pos_idx2 = pos_indices2[:pos_samples2]
            neg_idx2 = neg_indices2[:neg_samples2]
            idx2 = np.concatenate([pos_idx2, neg_idx2])
            new_samples2 = U_X[idx2]
            new_labels2 = np.array([1]*pos_samples2 + [0]*neg_samples2)
            # Remove selected samples from U_X
            mask = np.ones(len(U_X), dtype=bool)
            mask[idx2] = False
            U_X = U_X[mask]
            U_index = U_index[mask]

            # Add new samples to L1
            L1_X = np.vstack([L1_X, new_samples2])
            if self.view1_features is not None:
                new_samples2_view = new_samples2[:, self.view1_features]
                L1_X_view = np.vstack([L1_X_view, new_samples2_view])
            else:
                L1_X_view = np.vstack([L1_X_view, new_samples2])
            L1_y = np.concatenate([L1_y, new_labels2])

            # If no more samples to label, break
            if len(U_X) == 0:
                break

        # Final training
        self.estimator1.fit(L1_X_view, L1_y)
        self.estimator2.fit(L2_X_view, L2_y)

        return self

    def predict(self, X):
        if self.view1_features is not None:
            X_view1 = X[:, self.view1_features]
        else:
            X_view1 = X
        if self.view2_features is not None:
            X_view2 = X[:, self.view2_features]
        else:
            X_view2 = X

        # Predict with both estimators and combine predictions
        pred1 = self.estimator1.predict_proba(X_view1)
        pred2 = self.estimator2.predict_proba(X_view2)
        avg_pred = (pred1 + pred2) / 2
        return np.argmax(avg_pred, axis=1)

    def predict_proba(self, X):
        if self.view1_features is not None:
            X_view1 = X[:, self.view1_features]
        else:
            X_view1 = X
        if self.view2_features is not None:
            X_view2 = X[:, self.view2_features]
        else:
            X_view2 = X

        pred1 = self.estimator1.predict_proba(X_view1)
        pred2 = self.estimator2.predict_proba(X_view2)
        avg_pred = (pred1 + pred2) / 2
        return avg_pred

# semi-supervised
class SemiSupervisedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    An ensemble of semi-supervised classifiers that fits within the sklearn interface.
    """

    def __init__(self, base_estimator=None, n_estimators=10, voting='hard'):
        """
        Initializes the ensemble classifier.

        Parameters:
        - base_estimator: The base estimator to use for each ensemble member.
        - n_estimators: The number of estimators in the ensemble.
        - voting: 'hard' for majority voting, 'soft' for averaging predicted probabilities.
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.voting = voting  # 'hard' or 'soft'

    def fit(self, X, y):
        """
        Fits the ensemble of classifiers on the provided data.

        Parameters:
        - X: array-like of shape (n_samples, n_features), the training input samples.
        - y: array-like of shape (n_samples,), the target values with unlabeled samples marked as -1.

        Returns:
        - self: Fitted estimator.
        """
        # Validate the input data
        X, y = check_X_y(X, y, accept_sparse=True)
        self.classes_ = np.unique(y[y != -1])
        self.estimators_ = []

        # Create and fit each estimator in the ensemble
        for _ in range(self.n_estimators):
            estimator = clone(self.base_estimator)
            self_training_estimator = SelfTrainingClassifier(estimator)
            self_training_estimator.fit(X, y)
            self.estimators_.append(self_training_estimator)

        return self

    def predict(self, X):
        """
        Predicts class labels for samples in X.

        Parameters:
        - X: array-like of shape (n_samples, n_features), the input samples.

        Returns:
        - y_pred: array-like of shape (n_samples,), the predicted classes.
        """
        X = check_array(X, accept_sparse=True)

        if self.voting == 'hard':
            # Collect predictions from each estimator
            predictions = np.asarray([estimator.predict(X) for estimator in self.estimators_]).T
            # Majority vote
            y_pred = np.apply_along_axis(
                lambda x: np.bincount(x, minlength=len(self.classes_)).argmax(), axis=1, arr=predictions
            )
            return y_pred
        elif self.voting == 'soft':
            # Average predicted probabilities
            probas = np.asarray([estimator.predict_proba(X) for estimator in self.estimators_])
            avg_proba = np.mean(probas, axis=0)
            y_pred = self.classes_[np.argmax(avg_proba, axis=1)]
            return y_pred
        else:
            raise ValueError("Voting must be 'hard' or 'soft'")

    def predict_proba(self, X):
        """
        Predicts class probabilities for samples in X.

        Parameters:
        - X: array-like of shape (n_samples, n_features), the input samples.

        Returns:
        - avg_proba: array-like of shape (n_samples, n_classes), the class probabilities.
        """
        X = check_array(X, accept_sparse=True)
        probas = np.asarray([estimator.predict_proba(X) for estimator in self.estimators_])
        avg_proba = np.mean(probas, axis=0)
        return avg_proba


# AutoEncoder
class AutoencoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, hidden_layer_sizes=(32,), activation='relu', solver='adam', 
                 batch_size='auto', learning_rate_init=0.001, max_iter=200, random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self._is_fitted = False

    def fit(self, X, y=None):
        n_features = X.shape[1]
        # Create a symmetric autoencoder architecture
        hidden_sizes = list(self.hidden_layer_sizes)
        # Encoder
        encoder_layer_sizes = hidden_sizes
        # Decoder (reverse of encoder)
        decoder_layer_sizes = hidden_sizes[::-1]
        # Total layer sizes
        layer_sizes = encoder_layer_sizes + decoder_layer_sizes
        # Initialize the MLPRegressor
        self.autoencoder = MLPRegressor(
            hidden_layer_sizes=layer_sizes,
            activation=self.activation,
            solver=self.solver,
            batch_size=self.batch_size,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        # Fit the autoencoder to reconstruct the input
        self.autoencoder.fit(X, X)
        self._is_fitted = True
        return self

    def transform(self, X):
        check_is_fitted(self, '_is_fitted')
        # Compute the activations of each layer
        X_transformed = self._compute_hidden_activations(X)
        # Return the output of the last encoder layer (the bottleneck layer)
        # Assuming the encoder layers are the first half
        n_encoder_layers = len(self.hidden_layer_sizes)
        return X_transformed[n_encoder_layers - 1]

    def _compute_hidden_activations(self, X):
        # This method computes the activations at each hidden layer
        activations = [X]
        for i in range(len(self.autoencoder.coefs_)):
            activation = np.dot(activations[i], self.autoencoder.coefs_[i]) + self.autoencoder.intercepts_[i]
            if i < len(self.autoencoder.coefs_) - 1:
                activation = self._activation_function(activation)
            activations.append(activation)
        return activations[1:-1]  # Exclude input and output layers

    def _activation_function(self, X):
        # Apply the activation function
        if self.activation == 'identity':
            return X
        elif self.activation == 'logistic':
            return 1 / (1 + np.exp(-X))
        elif self.activation == 'tanh':
            return np.tanh(X)
        elif self.activation == 'relu':
            return np.maximum(0, X)
        else:
            raise ValueError(f"Unsupported activation function '{self.activation}'")

class SemiSupervisedAutoencoderClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier=None, hidden_layer_sizes=(32,), activation='relu', 
                 solver='adam', batch_size='auto', learning_rate_init=0.001, max_iter=200, 
                 random_state=None):
        self.base_classifier = base_classifier or SVC(probability=True, random_state=random_state)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self._is_fitted = False

    def fit(self, X, y):
        y = np.asarray(y)
        # Identify unlabeled data (label == -1 or None)
        unlabeled_mask = (y == -1) | (y == None)
        X_labeled = X[~unlabeled_mask]
        y_labeled = y[~unlabeled_mask]
        X_unlabeled = X[unlabeled_mask]
        # Combine all data for autoencoder training
        X_all = np.vstack([X_labeled, X_unlabeled])
        # Fit the autoencoder transformer
        self.autoencoder = AutoencoderTransformer(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            batch_size=self.batch_size,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.autoencoder.fit(X_all)
        # Transform data using the trained autoencoder
        X_labeled_encoded = self.autoencoder.transform(X_labeled)
        X_unlabeled_encoded = self.autoencoder.transform(X_unlabeled)
        X_encoded = np.vstack([X_labeled_encoded, X_unlabeled_encoded])
        # Prepare labels for semi-supervised learning
        y_full = np.concatenate([y_labeled, np.full(X_unlabeled.shape[0], -1)])
        # Train the semi-supervised classifier
        self.classifier = SelfTrainingClassifier(self.base_classifier)
        self.classifier.fit(X_encoded, y_full)
        self._is_fitted = True
        return self

    def predict(self, X):
        check_is_fitted(self, '_is_fitted')
        X_encoded = self.autoencoder.transform(X)
        return self.classifier.predict(X_encoded)

    def predict_proba(self, X):
        check_is_fitted(self, '_is_fitted')
        X_encoded = self.autoencoder.transform(X)
        return self.classifier.predict_proba(X_encoded)