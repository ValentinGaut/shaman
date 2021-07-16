# Copyright 2020 BULL SAS All rights reserved
"""
Tests that the initial parametrizations methods work properly.
"""
# Disable the could be a function for unit testing
# pylint: disable=no-self-use
# Disable name too longs (necessary for clarity in testing)
# pylint: disable=invalid-name

import unittest
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from pathlib import Path

from . import clean_input

from bbo.initial_parametrizations import (
    uniform_random_draw,
    latin_hypercube_sampling,
    hybrid_lhs_uniform_sampling,
)

from bbo.smart_initialization.initialisation_plan import (
    initialize_optimizer_from_model
)

from bbo.smart_initialization.data_matcher import DataMatcher


class TestUniformRandomDraws(unittest.TestCase):
    """
    Tests that the initial drawing of parameters work properly.
    """

    def setUp(self):
        """
        Sets up the testing procedure by setting the random seed of the project.
        """
        np.random.seed(2)
        CURRENT_DIR = Path(__file__).parent.resolve()
        DATA_PATH = CURRENT_DIR / "data" / "test_data.csv"
        df = pd.read_csv(DATA_PATH)
        df = clean_input(df)
        fakeapp_array = df[["scatter", "leads", "lead_advance"]].values
        sro_array = df[["SRO_CLUSTER_THRESHOLD", "SRO_DSC_BINSIZE", "SRO_PREFETCH_SIZE", "SRO_SEQUENCE_LENGTH"]].values

        matcher = DataMatcher()
        matcher.fit(fakeapp_array, sro_array)

        self.model = matcher

    def test_uniform_random_draw_array(self):
        """
        Tests that the uniform random draw functions behaves as expected when the parameter space
        is described by arrays.
        """
        number_of_parameters = 2
        parameter_space = np.array([[1, 2, 3], [4, 5, 6, 7], [8, 9, 10]])
        expected_result = np.array([[1, 4, 10], [2, 6, 8]])
        actual_result = uniform_random_draw(number_of_parameters, parameter_space)
        assert_array_equal(actual_result, expected_result)

    def test_uniform_random_draw_array_large(self):
        """
        Tests that the uniform random draw functions behaves as expected when the parameter space
        is described by arrays.
        """
        number_of_parameters = 5
        parameter_space = np.array([[1, 2, 3], [4, 5, 6, 7], [8, 9, 10]])
        expected_result = np.array(
            [[1, 7, 9], [2, 4, 10], [1, 7, 8], [3, 6, 8], [3, 5, 8]]
        )
        actual_result = uniform_random_draw(number_of_parameters, parameter_space)
        assert_array_equal(actual_result, expected_result)

    def test_uniform_random_draw_range(self):
        """
        Tests that the uniform random draw works as expected when given a range as a parametric
        space.
        """
        number_of_parameters = 2
        parameter_space = np.array(
            [np.arange(1, 10), np.arange(10, 20), np.arange(20, 30)]
        )
        expected_result = np.array([[9, 16, 28], [9, 12, 27]])
        actual_result = uniform_random_draw(number_of_parameters, parameter_space)
        assert_array_equal(actual_result, expected_result)

    def test_uniform_random_draw_except(self):
        """
        Tests that the uniform random draw works as expected when given a range 1D parametric
        space.
        """
        number_of_parameters = 2
        parameter_space = np.array([[[1, 2, 3], [4, 5, 6, 7], [8, 9, 10]]])
        expected_result = np.array([[1, 4, 10], [2, 6, 8]])
        actual_result = uniform_random_draw(number_of_parameters, parameter_space)
        assert_array_equal(actual_result, expected_result)

    def test_latin_hypercube_sampling(self):
        """
        Tests that the latin hypercube sampling works properly.
        """
        number_of_parameters = 4
        parameter_space = np.array([np.arange(1, 10), np.arange(1, 5)])
        actual_result = latin_hypercube_sampling(number_of_parameters, parameter_space)
        expected_result = np.array([[5.0, 1.0], [4.0, 2.0], [9.0, 3.0], [1.0, 4.0]])
        assert_array_equal(actual_result, expected_result)

    def test_latin_hypercube_sampling_too_many_parameters(self):
        """
        Tests that the latin hypercube sampling returns an Assertion Error when the user asks for
        more parameter draws than the size of the smallest dimension.
        """
        number_of_parameters = 6
        parameter_space = np.array([np.arange(1, 10), np.arange(1, 5)])
        with self.assertRaises(AssertionError):
            latin_hypercube_sampling(number_of_parameters, parameter_space)

    def test_non_collapse_property(self):
        """
        Tests that the non collapse property of the latin hypercube is respected, meaning that
        there is no duplicate column values.
        """
        np.random.seed(10)
        number_of_parameters = 4
        parameter_space = np.array([np.arange(1, 10), np.arange(1, 5)])
        actual_result = latin_hypercube_sampling(number_of_parameters, parameter_space)
        len_unique_values = [len(np.unique(axis)) for axis in actual_result.T]
        dim_size = [value == number_of_parameters for value in len_unique_values]
        self.assertTrue(all(dim_size), "Collapsible property was not respected.")

    def test_hybrid_lhs_uniform_sampling_small(self):
        """
        Tests that the hybride sampling works like the latin hypercube sampling if the number of
        parameter is less than the size of the smallest sample.
        """
        number_of_parameters = 4
        parameter_space = np.array([np.arange(1, 10), np.arange(1, 5)])
        actual_result = hybrid_lhs_uniform_sampling(
            number_of_parameters, parameter_space
        )
        np.random.seed(2)
        expected_result = latin_hypercube_sampling(
            number_of_parameters, parameter_space
        )
        assert_array_equal(actual_result, expected_result)

    def test_hybrid_lhs_uniform_sampling_large(self):
        """
        Tests that the latin hypercube sampling with random uniform works properly.
        """
        number_of_parameters = 8
        parameter_space = np.array([np.arange(1, 10), np.arange(1, 5)])
        actual_result = hybrid_lhs_uniform_sampling(
            number_of_parameters, parameter_space
        )
        np.random.seed(2)
        lhs = latin_hypercube_sampling(4, parameter_space)
        ur = uniform_random_draw(4, parameter_space)
        assert_array_equal(actual_result, np.append(lhs, ur, axis=0))

    def test_smart_init_load_model(self):
        """Test to check that the model for smart lhs
        is correctly loaded;
        """
        self.assertIsNotNone(self.model)

    def test_smart_init_predict(self):
        """Test that the prediction of the model
        of smart lhs is correct
        """
        fakeapp_test = [18874368, 286, 0]
        expected = [2.0000000e+00, 1.1272192e+07, 3.9321600e+06, 2.1000000e+02]
        res = list(self.model.predict([fakeapp_test])[0])
        self.assertEqual(res, expected)

    def test_smart_init_space(self):
        """Test that the initial points are
        properly generated using smart lhs
        """
        np.random.seed(10)
        fakeapp_test = [18874368, 286, 0]
        parameter_space = np.array([np.arange(2, 102, 20), np.arange(262144, 1048576*20, 1048576), np.arange(1048576, 10485760, 1048576), np.arange(50, 750, 100)])
        result = initialize_optimizer_from_model(4, parameter_space, self.model, fakeapp_test)
        expected = [[2.0, 11272192.0, 3932160.0, 210.0], [42, 6553600, 2097152, 50], [82, 18087936, 5242880, 650], [22, 262144, 9437184, 350]]
        self.assertEqual(result.tolist(), expected)

    def test_smart_init_not_in_space(self):
        """Test that the initialization raises an error
        if the predicted point is not in the space.
        """
        fakeapp_test = [18874368, 286, 0]
        parameter_space = np.array([np.arange(2, 102, 20), np.arange(262144, 1048576, 1048576), np.arange(1048576, 10485760, 1048576), np.arange(50, 750, 100)])
        with self.assertRaises(ValueError) as context:
            initialize_optimizer_from_model(4, parameter_space, self.model, fakeapp_test)
            self.assertTrue("Point not in space on dimension 1" in str(context.exception))


if __name__ == "__main__":
    unittest.main()
