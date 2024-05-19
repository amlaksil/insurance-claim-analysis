#!/usr/bin/python3
"""
This module is designed to perform analysis on insurance claim data. It
includes functionality to read the data from a specified file path, and
provides tools to visualize and analyze the data using pandas
matplotlib, and seaborn.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class InsuranceDataAnalysis:
    """
    A class to perform data analysis on insurance data.

    Attributes:
        file_path(str): The path ot the txt file containg insurance data.
        df (pd.dataframe): The dataframe containg the loaded data.
    """
    def __init__(self, file_path):
        """
        Constructs the necessary attributes for the InsuranceDataAnalysis
        object.

        Args:
            file_path(str): The file path of the CSV file containing insurance
        claim data.
        """
        self.df = None
        self.file_path = file_path  # This will invoke the setter method

    @property
    def file_path(self):
        """Returns the current file path."""
        return self._file_path

    @file_path.setter
    def file_path(self, new_file_path):
        """
        Sets a new file path for the data file and reloads the DataFrame.

        Args:
            new_file_path(str): The new file path of the txt file
        containing insurance claim data.
        """
        self._validate_file_path(new_file_path)
        self._file_path = new_file_path
        self.df = pd.read_csv(self._file_path, sep='|', low_memory=False)

    def _validate_file_path(self, file_path):
        """
        Validates if the file path exists and is a CSV file.

        Parameters
        ----------
        file_path (str):The file path to validate.

        Raises
        ------
        FileNotFoundError
            If the provided file path does not exist.
        ValueError
            If the provided file path is not a CSV file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file at {file_path} does not exist.")
        if not file_path.endswith('.txt'):
            raise ValueError("The file must be a txt file.")

    def data_summarization(self):
        """
        Perform data summarization on the DataFrame.

        Returns:
        - descriptive_stats (pandas.DataFrame): A DataFrame containing
        descriptive statistics of the data.
        - data_structure (pandas.Series): A Series containing the data types
        of each column in the DataFrame.
        """
        # Descriptive Statistics
        descriptive_stats = self.df.describe()

        # Data Structure
        data_structure = self.df.dtypes

        return descriptive_stats, data_structure

    def data_quality_assessment(self):
        """
        Perform a data quality assessment by checking for missing values in
        the DataFrame.

        Returns:
            - missing_values (pandas.Series): A Series containing the count
            of missing values for each column in the DataFrame.
        """
        # Check for missing values
        missing_values = self.df.isnull().sum()

        return missing_values

    def univariate_analysis(self, sample_size=10000, num_bins=30):
        """
        Perform univariate analysis by creating histograms for numerical
        columns and bar charts for categorical columns.

        Args:
        - sample_size (int): The number of samples to plot from the dataset.
        Default is 10,000.
        - num_bins (int): The number of bins for histograms. Default is 30.

        Returns:
        - univariate_plots (list): A list of matplotlib figures, each
        containing a histogram or bar chart.
        - univariate_data (dict): A dictionary containing the data used for
        each plot.
        """
        univariate_plots = []
        univariate_data = {}

        if self.df.empty:
            print("DataFrame is empty. No plots to generate.")
            return univariate_plots, univariate_data

        # Setting style for the plots
        sns.set(style="whitegrid")

        # Sampling the data if it's too large
        if len(self.df) > sample_size:
            df_sample = self.df.sample(n=sample_size, random_state=42)
        else:
            df_sample = self.df

        # Histograms for numerical columns
        for col in df_sample.select_dtypes(
                include=['int64', 'float64']).columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            # Drop NaN values before calculating the histogram
            non_nan_values = df_sample[col].dropna()
            counts, bin_edges = np.histogram(non_nan_values, bins=num_bins)
            sns.histplot(
                non_nan_values, bins=num_bins,
                kde=True, color='skyblue', ax=ax)
            ax.set_title(f'Histogram of {col}', fontsize=15)
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            univariate_plots.append(fig)

            # Extract histogram data
            univariate_data[col] = {
                'type': 'histogram', 'counts': counts, 'bin_edges': bin_edges}

            plt.close(fig)  # Close the figure after adding it to the list

        # Bar charts for categorical columns
        for col in df_sample.select_dtypes(include=['object']).columns:
            fig, ax = plt.subplots(figsize=(16, 10))
            # Only plot top 20 most common categories to avoid clutter
            top_categories = df_sample[col].value_counts().nlargest(20)
            sns.barplot(
                    x=top_categories.index, y=top_categories.values,
                    hue=top_categories.index, palette='viridis',
                    ax=ax, dodge=False, legend=False)
            ax.set_title(f'Bar chart of {col}', fontsize=15)
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel('Count', fontsize=12)

            # Rotate x-axis labels if they are too long
            if len(max(top_categories.index, key=len)) > 10:
                plt.xticks(rotation=25, ha='right')

            univariate_plots.append(fig)

            # Extract bar chart data
            univariate_data[col] = {
                'type': 'bar_chart',
                'categories': top_categories.index.tolist(),
                'counts': top_categories.values.tolist()
            }

            plt.close(fig)  # Close the figure after adding it to the list

        return univariate_plots, univariate_data

    def sample_data(self, sample_fraction=0.1, stratify_col=None):
        """
        Sample a fraction of the data.

        Args:
        - sample_fraction: float, fraction of data to sample.
        - stratify_col: str, column to stratify by (optional).

        Returns:
        - sampled_df: DataFrame, sampled subset of the data.
        """
        if stratify_col:
            sampled_df = self.df.groupby(stratify_col, group_keys=False).apply(
                lambda x: x.sample(frac=sample_fraction))
        else:
            sampled_df = self.df.sample(frac=sample_fraction)
        return sampled_df

    def bivariate_multivariate_analysis_by_postalcode(
            self, sample_fraction=0.1, stratify_col='PostalCode'):
        """
        Explore the relationships between the monthly changes in TotalPremium
        and TotalClaims as a function of PostalCode using a sampled subset.

        Args:
        - sample_fraction: float, fraction of data to sample.
        - stratify_col: str, column to stratify by.

        Returns:
        - correlation_matrix_by_postalcode: Dictionary, correlation matrices
        for each PostalCode.
        - scatter_plot_data_by_postalcode: Dictionary, data used for scatter
        plots for each PostalCode.
        """
        sampled_df = self.sample_data(sample_fraction, stratify_col)

        sampled_df['TransactionMonth'] = pd.to_datetime(
            sampled_df['TransactionMonth'])
        sampled_df['TotalPremium_MonthlyChange'] = sampled_df.groupby(
            'PostalCode')['TotalPremium'].diff()
        sampled_df['TotalClaims_MonthlyChange'] = sampled_df.groupby(
            'PostalCode')['TotalClaims'].diff()
        sampled_df.dropna(
            subset=['TotalPremium_MonthlyChange', 'TotalClaims_MonthlyChange'],
            inplace=True)

        grouped = sampled_df.groupby('PostalCode')
        correlation_matrix_by_postalcode = {}
        scatter_plot_data_by_postalcode = {}

        for postalcode, group in grouped:
            correlation_matrix = group[
                ['TotalPremium_MonthlyChange', 'TotalClaims_MonthlyChange']
            ].corr()
            correlation_matrix_by_postalcode[postalcode] = correlation_matrix
            scatter_plot_data_by_postalcode[postalcode] = group[
                ['TotalPremium_MonthlyChange', 'TotalClaims_MonthlyChange']]

        return (
            correlation_matrix_by_postalcode,
            scatter_plot_data_by_postalcode
        )

    def plot_scatter_plots(self, scatter_plot_data_by_postalcode):
        """
        Create scatter plots from the scatter plot data for each PostalCode.

        Args:
        - scatter_plot_data_by_postalcode: Dictionary, data used for scatter
        plots for each PostalCode.
        """
        for postalcode, data in scatter_plot_data_by_postalcode.items():
            sns.scatterplot(
                x='TotalPremium_MonthlyChange',
                y='TotalClaims_MonthlyChange', data=data)
            plt.title(
                'Scatter Plot of Monthly Changes in TotalPremium vs' +
                f'TotalClaims for PostalCode {postalcode}')
            plt.xlabel('Monthly Change in TotalPremium')
            plt.ylabel('Monthly Change in TotalClaims')
            plt.show()

    def data_comparison_trends_over_geography(self):
        """
        Compare trends over geography for insurance cover type, premium
        auto make, etc.

        Returns:
            list: A list of matplotlib figures, each containing a line
            plot showing the trend over geography for a specific attribute
            (e.g., TotalPremium, AutoMake) based on postal codes.
        """
        comparison_plots = []

        # Example: Trends Over Geography for TotalPremium, AutoMake, etc.
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            x='PostalCode', y='TotalPremium', hue='ItemType', data=self.df)
        plt.title('Trends Over Geography for TotalPremium by ItemType')
        plt.xticks(rotation=45)
        comparison_plots.append(plt.gcf())

        return comparison_plots

    def outlier_detection(self):
        """
        Detects and visualizes outliers in numerical columns of the DataFrame.

        This method creates a box plot for each numerical column to visually
        inspect for outliers. It then identifies columns with outliers using
        the IQR method and returns the columns and their corresponding
        outlier data.

        Returns:
            tuple:
                - outlier_plots (list): List of columns for which box plots
                were created.
                - outlier_columns_df (pd.DataFrame): DataFrame containing only
        the columns with detected outliers.
        """
        outlier_plots = []
        outlier_columns = []
        for column in self.df.select_dtypes(include=['int64', 'float64']):
            _, ax = plt.subplots(figsize=(8, 6))
            ax.boxplot(self.df[column], vert=False)
            ax.set_title(f'Box plot for {column}', fontsize=14)
            ax.set_xlabel('Value', fontsize=12)
            ax.set_yticklabels([column], fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.show()
            outlier_plots.append(column)
            # Check for outliers and store column name if outliers are found
            if self.has_outliers(self.df[column]):
                outlier_columns.append(column)
        return outlier_plots, self.df[outlier_columns]

    def has_outliers(self, series, k=1.5):
        """
        Determines if a numerical series has outliers using the IQR method.

        This method calculates the first and third quartiles, the IQR, and the
        lower and upper bounds.
        It then checks if any values in the series lie outside these bounds
        indicating the presence of outliers.

        Args:
            series (pd.Series): The numerical series to check for outliers.
            k (float, optional): The multiplier for the IQR to determine the
        bounds. Default is 1.5.

        Returns:
            bool: True if outliers are present, False otherwise.
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        return any((series < lower_bound) | (series > upper_bound))

    def visualization(self):
        """
        Generate three plots to visualize key insights from the insurance data.

        - Trend of TotalPremium and TotalClaims over Time
        - Distribution of TotalPremium vs. TotalClaims
        - Correlation Heatmap of selected numerical features

        Returns:
        str: Message indicating that the visualization plots have been saved.
        """
        # Trend of TotalPremium and TotalClaims over Time
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            x='TransactionMonth', y='TotalPremium',
            data=self.df, label='TotalPremium', marker='o')
        sns.lineplot(
            x='TransactionMonth', y='TotalClaims',
            data=self.df, label='TotalClaims', marker='o')
        plt.title(
            'Trend of TotalPremium and TotalClaims over Time', fontsize=15)
        plt.xlabel('TransactionMonth', fontsize=12)
        plt.ylabel('Amount', fontsize=12)
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.legend()
        plt.tight_layout()  # Adjust layout to prevent clipping of labels
        plt.savefig('TotalPremium_TotalClaims_Trend.png')
        plt.close()

        # Distribution of TotalPremium vs. TotalClaims
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='TotalPremium', y='TotalClaims', data=self.df)
        plt.title('Distribution of TotalPremium vs. TotalClaims', fontsize=15)
        plt.xlabel('TotalPremium', fontsize=12)
        plt.ylabel('TotalClaims', fontsize=12)
        plt.savefig('TotalPremium_TotalClaims_Distribution.png')
        plt.close()

        # Correlation Heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.df[[
            'TotalPremium', 'TotalClaims', 'RegistrationYear', 'Cylinders',
            'SumInsured']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap', fontsize=15)
        plt.savefig('Correlation_Heatmap.png')
        plt.close()

        str_f = 'Visualization plots saved as TotalPremium_TotalClaims_' +\
            'Trend.png, TotalPremium_TotalClaims_Distribution.png, and' +\
            ' Correlation_Heatmap.png'
        return str_f
