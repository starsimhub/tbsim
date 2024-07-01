import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from unittest.mock import patch

# Assuming the functions are in a module named 'plots'
from plots import plot_epi, plot_hh, plot_nut, plot_active_infections

class TestMyModule(unittest.TestCase):

    def setUp(self):
        # Create a sample dataframe for testing
        self.df_epi = pd.DataFrame({
            'year': [1942, 1943, 1944],
            'rand_seed': [0, 0, 0],
            'arm': ['A', 'A', 'A'],
            'channel1': [10, 15, 20],
            'channel2': [5, 10, 15]
        })

        self.df_hh = pd.DataFrame({
            'HH Size': [1, 2, 3],
            'rand_seed': [0, 0, 0],
            '1942': [5, 10, 15],
            '1943': [3, 6, 9]
        }).set_index('HH Size')

        self.df_nut = pd.DataFrame({
            'Arm': ['VITAMIN', 'VITAMIN', 'CONTROL', 'CONTROL'],
            'Macro': [1.0, 2.0, 3.0, 4.0],
            'Micro': [1.0, 2.0, 3.0, 4.0],
            'rand_seed': [0, 0, 0, 0],
            '1942': [5.0, 10.0, 15.0, 20.0],
            '1943': [3.0, 6.0, 9.0, 12.0]
        })
        
        
        self.df_active_infections = pd.DataFrame({
            'arm': ['VITAMIN', 'VITAMIN', 'CONTROL', 'CONTROL'],
            'rand_seed': [0, 0, 0, 0],
            'year': [1942, 1943, 1942, 1943],
            'cum_active_infections': [10, 15, 20, 25]
        })

    @patch('plots.sc.savefig')
    def test_plot_epi(self, mock_savefig):
        plot_epi(self.df_epi)
        mock_savefig.assert_called_once()
        plt.show()
        plt.close('all')  # Close plots to prevent resource warning

    @patch('plots.sc.savefig')
    def test_plot_hh(self, mock_savefig):
        plot_hh(self.df_hh)
        mock_savefig.assert_called_once()
        plt.show()
        plt.close('all')  # Close plots to prevent resource warning

    @patch('plots.sc.savefig')
    def test_plot_nut(self, mock_savefig):
        
        # index = pd.MultiIndex.from_tuples([
        #     ('CONTROL', 'MARGINAL', 'DEFICIENT'),
        #     ('CONTROL', 'MARGINAL', 'NORMAL'),
        #     ('CONTROL', 'SLIGHTLY_BELOW_STANDARD', 'DEFICIENT'),
        #     ('CONTROL', 'STANDARD_OR_ABOVE', 'NORMAL'),
        #     ('VITAMIN', 'SLIGHTLY_BELOW_STANDARD', 'DEFICIENT'),
        #     ('VITAMIN', 'SLIGHTLY_BELOW_STANDARD', 'NORMAL'),
        #     ('VITAMIN', 'STANDARD_OR_ABOVE', 'NORMAL'),
        #     ('VITAMIN', 'UNSATISFACTORY', 'DEFICIENT'),
        # ], names=['Arm', 'Macro', 'Micro'])

        # data = {
        #     '1942': [76.0, 35.0, 33.0, 127.0, 98.0, 21.0, 90.0, 98.0 ],
        #     '1944': [36.0, 4.0, 46.0, 33.0, 210.0, 13.0, 82.0, 258.0],
        #     'rand_seed': [0, 0, 0, 0, 0, 4, 4, 4 ]
        # }

        data={
                'Arm': ['CONTROL', 'CONTROL', 'CONTROL', 'CONTROL', 'CONTROL', 'CONTROL', 'VITAMIN', 'VITAMIN', 'VITAMIN', 'VITAMIN',
                        'VITAMIN', 'VITAMIN', 'CONTROL', 'CONTROL', 'CONTROL', 'CONTROL', 'CONTROL', 'CONTROL', 'VITAMIN', 'VITAMIN',
                        'VITAMIN', 'VITAMIN', 'VITAMIN', 'VITAMIN', 'CONTROL', 'CONTROL', 'CONTROL', 'CONTROL', 'CONTROL', 'CONTROL',
                        'VITAMIN', 'VITAMIN', 'VITAMIN', 'VITAMIN', 'VITAMIN', 'VITAMIN', 'CONTROL', 'CONTROL', 'CONTROL', 'CONTROL',
                        'CONTROL', 'CONTROL', 'VITAMIN', 'VITAMIN', 'VITAMIN', 'VITAMIN', 'VITAMIN', 'VITAMIN', 'CONTROL', 'CONTROL',
                        'CONTROL', 'CONTROL', 'CONTROL', 'CONTROL', 'VITAMIN', 'VITAMIN', 'VITAMIN', 'VITAMIN', 'VITAMIN', 'VITAMIN',
                        'CONTROL', 'CONTROL'],
                'Macro': ['MARGINAL', 'MARGINAL', 'SLIGHTLY_BELOW_STANDARD', 'SLIGHTLY_BELOW_STANDARD', 'STANDARD_OR_ABOVE', 
                        'UNSATISFACTORY', 'MARGINAL', 'MARGINAL', 'SLIGHTLY_BELOW_STANDARD', 'SLIGHTLY_BELOW_STANDARD', 
                        'STANDARD_OR_ABOVE', 'UNSATISFACTORY', 'STANDARD_OR_ABOVE', 'MARGINAL', 'MARGINAL', 'SLIGHTLY_BELOW_STANDARD', 
                        'SLIGHTLY_BELOW_STANDARD', 'STANDARD_OR_ABOVE', 'MARGINAL', 'MARGINAL', 'SLIGHTLY_BELOW_STANDARD', 
                        'SLIGHTLY_BELOW_STANDARD', 'STANDARD_OR_ABOVE', 'UNSATISFACTORY', 'STANDARD_OR_ABOVE', 'MARGINAL', 'MARGINAL', 
                        'SLIGHTLY_BELOW_STANDARD', 'SLIGHTLY_BELOW_STANDARD', 'STANDARD_OR_ABOVE', 'UNSATISFACTORY', 'MARGINAL', 
                        'MARGINAL', 'SLIGHTLY_BELOW_STANDARD', 'SLIGHTLY_BELOW_STANDARD', 'STANDARD_OR_ABOVE', 'UNSATISFACTORY', 
                        'STANDARD_OR_ABOVE', 'MARGINAL', 'MARGINAL', 'SLIGHTLY_BELOW_STANDARD', 'SLIGHTLY_BELOW_STANDARD', 
                        'STANDARD_OR_ABOVE', 'UNSATISFACTORY', 'MARGINAL', 'MARGINAL', 'SLIGHTLY_BELOW_STANDARD', 'SLIGHTLY_BELOW_STANDARD', 
                        'STANDARD_OR_ABOVE', 'UNSATISFACTORY', 'STANDARD_OR_ABOVE', 'MARGINAL'],
                'Micro': ['DEFICIENT', 'NORMAL', 'DEFICIENT', 'NORMAL', 'NORMAL', 'DEFICIENT', 'DEFICIENT', 'NORMAL', 'DEFICIENT', 'NORMAL',
                        'NORMAL', 'DEFICIENT', 'DEFICIENT', 'DEFICIENT', 'NORMAL', 'DEFICIENT', 'NORMAL', 'NORMAL', 'DEFICIENT', 'NORMAL',
                        'DEFICIENT', 'NORMAL', 'NORMAL', 'DEFICIENT', 'DEFICIENT', 'DEFICIENT', 'NORMAL', 'DEFICIENT', 'NORMAL', 'NORMAL',
                        'DEFICIENT', 'DEFICIENT', 'NORMAL', 'DEFICIENT', 'NORMAL', 'NORMAL', 'DEFICIENT', 'DEFICIENT', 'DEFICIENT', 'NORMAL',
                        'DEFICIENT', 'NORMAL', 'NORMAL', 'DEFICIENT', 'DEFICIENT', 'NORMAL', 'DEFICIENT', 'NORMAL', 'NORMAL', 'DEFICIENT',
                        'DEFICIENT', 'DEFICIENT', 'NORMAL', 'DEFICIENT', 'NORMAL', 'NORMAL', 'DEFICIENT'],
                '1942': [76.0, 35.0, 33.0, 127.0, 98.0, 58.0, 48.0, 23.0, 29.0, 99.0, 83.0, 76.0, np.nan, 96.0, 35.0, 26.0, 62.0, 70.0,
                        29.0, 84.0, 32.0, 33.0, 123.0, 102.0, 64.0, np.nan, 140.0, 48.0, 35.0, 76.0, 65.0, 36.0, 42.0, 15.0, 44.0, 101.0,
                        92.0, 33.0, np.nan, 143.0, 55.0, 23.0, 60.0, 54.0, 51.0, 68.0, 25.0, 33.0, 115.0, 136.0, 32.0, np.nan, 109.0, 38.0,
                        20.0, 75.0, 91.0, 82.0, 103.0, 38.0, 21.0, 90.0, 98.0, 44.0, np.nan],
                '1944': [36.0, 4.0, 46.0, 33.0, 210.0, 12.0, np.nan, 35.0, np.nan, 70.0, 213.0, 21.0, 69.0, 25.0, 4.0, 43.0, 25.0, 131.0,
                        6.0, 1.0, 28.0, np.nan, 70.0, 295.0, 25.0, 67.0, 27.0, 5.0, 65.0, 18.0, 155.0, 4.0, 1.0, 17.0, np.nan, 34.0, 240.0,
                        8.0, 106.0, 40.0, 10.0, 61.0, 21.0, 131.0, 15.0, np.nan, 16.0, np.nan, 57.0, 312.0, 8.0, 93.0, 46.0, 4.0, 52.0, 24.0,
                        162.0, 26.0, 1.0, 31.0, np.nan, 82.0, 258.0, 11.0, 83.0],
                'rand_seed': [0]*13 + [1]*13 + [2]*13 + [3]*13 + [4]*13
            }

        # Create the DataFrame
        df = pd.DataFrame(data)
        
        
        plot_nut(df)
        mock_savefig.assert_called_once()
        plt.show()
        plt.close('all')  # Close plots to prevent resource warning

    @patch('plots.sc.savefig')
    def test_plot_active_infections(self, mock_savefig):
        plot_active_infections(self.df_active_infections)
        mock_savefig.assert_called_once()
        plt.show()
        plt.close('all')  # Close plots to prevent resource warning

if __name__ == '__main__':
    unittest.main()
