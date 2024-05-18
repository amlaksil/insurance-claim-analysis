#!/usr/bin/python3
"""
Module for testing the InsuranceDataAnalysis class.
This module includes tests for documentation, PEP8 compliance, and
various functionalities of the InsuranceDataAnalysis class.
"""
import inspect
import unittest
from unittest.mock import patch, Mock
import matplotlib.pyplot as plt
import pandas as pd
import pycodestyle
from src.exploratory_data_analysis import InsuranceDataAnalysis
import src
MODULE_DOC = src.exploratory_data_analysis.__doc__


class TestInsuranceDataAnalysisDocs(unittest.TestCase):
    """
    Tests to check the documentation and style of InsuranceDataAnalysis
    class.
    """
    def setUp(self):
        """Set up for docstring tests"""
        self.base_funcs = inspect.getmembers(
            InsuranceDataAnalysis, inspect.isfunction)

    def test_pep8_conformance(self):
        """Test that src/exploratory_data_analysis.py conforms to PEP8."""
        for path in ['src/exploratory_data_analysis.py',
                     'tests/test_exploratory_data_analysis.py']:
            with self.subTest(path=path):
                errors = pycodestyle.Checker(path).check_all()
                self.assertEqual(errors, 0)

    def test_module_docstring(self):
        """Test for the existence of module docstring"""
        self.assertIsNot(MODULE_DOC, None,
                         "exploratory_data_analysis.py needs a docstring")
        self.assertTrue(len(MODULE_DOC) > 1,
                        "exploratory_data_analysis.py needs a docstring")

    def test_class_docstring(self):
        """Test for the InsuranceDataAnalysis class docstring"""
        self.assertIsNot(InsuranceDataAnalysis.__doc__, None,
                         "InsuranceDataAnalysis class needs a docstring")
        self.assertTrue(len(InsuranceDataAnalysis.__doc__) >= 1,
                        "InsuranceDataAnalysis class needs a docstring")

    def test_func_docstrings(self):
        """
        Test for the presence of docstrings in InsuranceDataAnalysis methods.
        """
        for func in self.base_funcs:
            with self.subTest(function=func):
                self.assertIsNot(
                    func[1].__doc__,
                    None,
                    f"{func[0]} method needs a docstring"
                )
                self.assertTrue(
                    len(func[1].__doc__) > 1,
                    f"{func[0]} method needs a docstring"
                )


class TestInsuranceDataAnalysis(unittest.TestCase):
    """
    This class contains unit tests for the methods of the
    `InsuranceDataAnalysis` class, which is responsible for
    performing exploratory data analysis on insurance data.
    Each test method focuses on a specific aspect of the data
    analysis process, including data summarization, quality
    assessment, univariate analysis, bivariate and multivariate
    analysis, outlier detection, and visualization.
    """
    def setUp(self):
        """
        Set up an instance of InsuranceDataAnalysis before each test.
        """
        self.file_path = 'data/first_10_rows.txt'
        self.analysis = InsuranceDataAnalysis(self.file_path)

    def test_data_summarization(self):
        """
        Test the data summarization method.
        """
        descriptive_stats, data_structure = self.analysis.data_summarization()

        self.assertTrue(descriptive_stats is not None)
        self.assertTrue(data_structure is not None)
        self.assertIsInstance(descriptive_stats, pd.core.frame.DataFrame)
        self.assertIsInstance(data_structure, pd.core.series.Series)

    def test_data_quality_assessment(self):
        """
        Test the data quality assessment method.
        """
        missing_values = self.analysis.data_quality_assessment()

        self.assertTrue(missing_values is not None)
        self.assertIsInstance(missing_values, pd.core.series.Series)

        self.assertEqual(missing_values['PolicyID'], 0)
        self.assertEqual(missing_values['Title'], 0)

        self.assertEqual(missing_values['NewVehicle'], 0)
        self.assertEqual(missing_values['NumberOfVehiclesInFleet'], 30)

        self.assertEqual(missing_values['CrossBorder'], 30)
        self.assertEqual(missing_values['Rebuilt'], 24)

    def test_univariate_analysis(self):
        """
        Test the univariate analysis method.
        """
        univariate_plots, univariate_data = self.analysis.univariate_analysis()
        self.assertTrue(univariate_plots is not None)
        self.assertTrue(univariate_data is not None)

        self.assertIsInstance(univariate_plots, list)
        self.assertIsInstance(univariate_data, dict)

        self.assertTrue(all(
            isinstance(plot, plt.Figure) for plot in univariate_plots))

    def test_sample_data(self):
        """
        Test the sample data method.
        """
        sample_data = self.analysis.sample_data()
        self.assertTrue(sample_data is not None)

        self.assertIsInstance(sample_data, pd.core.frame.DataFrame)

    def test_bivariate_multivariate_analysis(self):
        """
        Test the bivariate and multivariate analysis method.
        """
        correlation_matrix_by_postalcode, scatter_plot_data_by_postalcode = \
            self.analysis.bivariate_multivariate_analysis_by_postalcode()

        self.assertTrue(correlation_matrix_by_postalcode is not None)
        self.assertTrue(scatter_plot_data_by_postalcode is not None)

        self.assertIsInstance(correlation_matrix_by_postalcode, dict)
        self.assertIsInstance(scatter_plot_data_by_postalcode, dict)
        self.assertEqual(correlation_matrix_by_postalcode, {})
        self.assertEqual(scatter_plot_data_by_postalcode, {})

    def test_plot_scatter_plots(self):
        """
        Test the plot scatter plots method.
        """
        self.analysis.plot_scatter_plots({})

    @patch('matplotlib.pyplot.figure')
    @patch('seaborn.lineplot')
    @patch('matplotlib.pyplot.xticks')
    def test_data_comparison_trends_over_geography(
            self, mock_xticks, mock_lineplot, mock_figure):
        """
        Test the data comparison trends over geography method.
        """
        # Create a mock figure
        mock_fig = Mock()

        # Patch the return value of 'matplotlib.pyplot.figure'
        # to return the mock figure
        mock_figure.return_value = mock_fig

        # Mock xticks to return None (or any other mock object that
        # behaves like an iterable)
        mock_xticks.return_value = None

        comparison_plots = \
            self.analysis.data_comparison_trends_over_geography()

        mock_figure.assert_called()
        mock_lineplot.assert_called()
        mock_xticks.assert_called()

        self.assertTrue(comparison_plots is not None)
        self.assertIsInstance(comparison_plots, list)

    @patch('matplotlib.pyplot.figure')
    def test_outlier_detection(self, mock_figure):
        """
        Test the outlier detection method.
        """
        outlier_plots, outlier_columns_df = self.analysis.outlier_detection()

        mock_figure.assert_called()
        self.assertTrue(outlier_plots is not None)
        self.assertTrue(outlier_columns_df is not None)

        self.assertEqual(
            ['UnderwrittenCoverID', 'PolicyID', 'PostalCode',
             'mmcode', 'RegistrationYear', 'Cylinders', 'cubiccapacity',
             'kilowatts', 'NumberOfDoors', 'CustomValueEstimate',
             'CapitalOutstanding', 'CrossBorder', 'NumberOfVehiclesInFleet',
             'SumInsured', 'CalculatedPremiumPerTerm',
             'TotalPremium', 'TotalClaims'], outlier_plots)
        self.assertIsInstance(outlier_plots, list)
        self.assertIsInstance(outlier_columns_df, pd.core.frame.DataFrame)

    # @patch('src.exploratory_data_analysis.plt.savefig')
    # @patch('src.exploratory_data_analysis.plt.close')
    @patch('src.exploratory_data_analysis.sns.scatterplot')
    @patch('src.exploratory_data_analysis.sns.heatmap')
    @patch('src.exploratory_data_analysis.sns.lineplot')
    def test_visualization(
            self, mock_lineplot, mock_heatmap, mock_scatterplot):
        """
        #mock_close, mock_savefig):
        Test the visualization method.
        """
        # Run the visualization method
        result = self.analysis.visualization()

        # Check if the plots were saved successfully
        self.assertEqual(
                result, 'Visualization plots saved as TotalPremium' +
                '_TotalClaims_Trend.png, TotalPremium_TotalClaims_' +
                'Distribution.png, and Correlation_Heatmap.png')
        # Check if the plotting functions were called
        self.assertTrue(mock_lineplot.called)
        self.assertTrue(mock_heatmap.called)
        self.assertTrue(mock_scatterplot.called)

        # Check if the savefig and close functions were called
        # self.assertTrue(mock_savefig.called)
        # self.assertTrue(mock_close.called)

    def tearDown(self):
        """
        Teardown method to clean up after each test, if necessary.
        """


if __name__ == '__main__':
    unittest.main()
