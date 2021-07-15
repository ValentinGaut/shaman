import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler


class DataMatcher:
    """This class is a tool used to predict a parametrization for the SRO
        using a 1NN approach
    """

    def fit(self, fakeapp_parameters: np.ndarray,
            sro_parameters: np.ndarray) -> MinMaxScaler:
        """Fits the model on the given data by fitting a MinMaxScaler and storing the
        normalized data matrix

        Args:
            fakeapp_parameters (2D ndarray): Array of shape
                (n_samples, n_features_fakeapp) describing the fakeapps used
                to train the model.
            sro_parameters (2D ndarray): Array of shape
                (n_samples, n_features_sro) describing the optimal sro
                parameters for the fakeapps used to train the model.
        """
        self.scaler = MinMaxScaler().partial_fit(fakeapp_parameters)
        self.normalized_fakeapp_parameters = (
            self.scaler.transform(fakeapp_parameters)
            )
        self.sro_parameters = sro_parameters
        return self.scaler

    def get_nearest_parametrization(self, fakeapp: np.ndarray,
                                    dataset: np.ndarray,
                                    sro: np.ndarray) -> np.ndarray:
        """Takes a decription vector and returns the parameters of the closest
        vector in dataset.

        Args:
            fakeapp (np.ndarray): description vector we want to find the
                closest neighbour of.
            dataset (np.ndarray): Dataset in which we are looking for the
                closest neighbour.
            sro (np.ndarray): Dataset containing the sro parameters of the
                description vectors.

        Returns:
            np.ndarray: A vector of sro associated to the closest description
                vector.
        """
        dist = euclidean_distances(fakeapp, dataset)
        res = sro[dist.argmin()]
        return res

    def predict(self, fakeapp_parameters: np.ndarray) -> np.ndarray:
        """Given a list of fakeapps parmeters, predicts sro parameters for
        each application.

        Args:
            fakeapp_parameters (np.ndarray): List of fakeapp parameters we
                want to make predictions on.

        Returns:
            np.ndarray: List of sro parameters predicted for each application.
        """
        scaled_fakeapps = self.scaler.transform(fakeapp_parameters)
        predictions = []
        for app in scaled_fakeapps:
            predictions.append(self.get_nearest_parametrization(
                [app], self.normalized_fakeapp_parameters, self.sro_parameters)
                )
        return np.array(predictions)
