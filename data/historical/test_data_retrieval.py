import unittest
from unittest.mock import patch
import pandas as pd
from fetch_data import fetch_historical_data  # Adjust the import according to your project structure

class TestDataRetrieval(unittest.TestCase):
    
    @patch('fetch_data.requests.get')
    def test_fetch_historical_data(self, mock_get):
        """
        Test fetching of historical data.
        """
        # Sample response data (as per your API response structure)
        mock_response_data = [
            {
                'period_id': '1DAY',  # Change period as needed (1DAY, 1HRS, etc.)
                "time_exchange": "2023-10-13T09:36:10.058Z",
                "time_coinapi": "2023-10-13T09:36:10.058Z",
                'limit': 1000,  # Limit the number of returned results
            },
        ]
        
        # Mock API response
        mock_get.return_value.json.return_value = mock_response_data
        mock_get.return_value.raise_for_status.return_value = None  # No exception
        
        # Call function
        df = fetch_historical_data('C06C0BB9-CEE4-4328-92B6-BE96DA8155E2')
        
        # Check DataFrame shape and column types
        self.assertEqual(df.shape, (len(mock_response_data), len(mock_response_data[0])))
        self.assertIsInstance(df.index, pd.DatetimeIndex)

# Run the tests
if __name__ == "__main__":
    unittest.main()
