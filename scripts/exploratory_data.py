#!/usr/bin/python3
from sys import argv, exit
from src.exploratory_data_analysis import InsuranceDataAnalysis
import pandas as pd
if __name__ == '__main__':
    if len(argv) == 1:
        print('Please provide file path')
        exit()
    data_analysis = InsuranceDataAnalysis(argv[1])
    descriptive_stats, data_structure = data_analysis.data_summarization()
    print(f'--Descriptive Stats--\n{descriptive_stats}\n--Data Structure--\n{data_structure}')

    missing_values = self.data_analysis.data_quality_assessment()
    print(f'--Missing Values--\n{missing_values}')

    univariate_plots, _ = data_analysis.univariate_analysis()
    for i, fig in enumerate(univariate_plots):
        fig.savefig(f'plot_{i}.png')
