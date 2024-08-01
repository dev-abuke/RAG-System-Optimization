import pandas as pd
from unittest.mock import MagicMock, patch
from scripts.testset_generator import generate_testset

def test_generate_testset():
    # Test default test_size
    with patch('scripts.testset_generator.load_dataset') as mock_load_dataset:
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = [
            {'article': 'Article 1', 'id': 1},
            {'article': 'Article 2', 'id': 2},
            {'article': 'Article 3', 'id': 3},
        ]
        mock_load_dataset.return_value = mock_dataset
        result = generate_testset()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 6

    # Test custom test_size
    with patch('scripts.testset_generator.load_dataset') as mock_load_dataset:
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = [
            {'article': 'Article 1', 'id': 1},
            {'article': 'Article 2', 'id': 2},
            {'article': 'Article 3', 'id': 3},
        ]
        mock_load_dataset.return_value = mock_dataset
        result = generate_testset(test_size=3)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    # Test saving to file
    with patch('scripts.testset_generator.load_dataset') as mock_load_dataset:
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = [
            {'article': 'Article 1', 'id': 1},
            {'article': 'Article 2', 'id': 2},
            {'article': 'Article 3', 'id': 3},
        ]
        mock_load_dataset.return_value = mock_dataset
        with patch('scripts.testset_generator.pd.Timestamp.now') as mock_timestamp:
            mock_timestamp.return_value = '20220101-000000'
            with patch('scripts.testset_generator.pd.DataFrame.to_csv') as mock_to_csv:
                generate_testset()
                mock_to_csv.assert_called_once_with(
                    '/teamspace/studios/this_studio/RAG-System-Optimization/data/20220101-000000testset.csv',
                    index=False
                )