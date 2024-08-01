import unittest
from unittest.mock import patch, mock_open
import yaml, os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.generate_test_name import load_config, get_test_name

class TestGenerateTestName(unittest.TestCase):

    @patch('scripts.generate_test_name.yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open, read_data="model: gpt-3.5-turbo\nretriever: hybrid\ntext_splitter: recursive\nchunk_size: 900\nchunk_overlap: 50\nquery_translation: hyde")
    def test_load_config(self, mock_open, mock_safe_load):
        """Test loading the config file."""
        mock_config = {'model': 'gpt-3.5-turbo', 'retriever': 'hybrid', 'text_splitter': 'recursive', 'chunk_size': 900, 'chunk_overlap': 50, 'query_translation': 'hyde'}
        mock_safe_load.return_value = mock_config
        config = load_config()
        self.assertEqual(config, mock_config)
        mock_safe_load.assert_called_once_with(mock_open.return_value)

    @patch('scripts.generate_test_name.load_config')
    def test_get_test_name(self, mock_load_config):
        """Test generating the test name."""
        mock_config = {'model': 'gpt-3.5-turbo', 'retriever': 'hybrid', 'text_splitter': 'recursive', 'chunk_size': 900, 'chunk_overlap': 50, 'query_translation':'hyde'}
        mock_load_config.return_value = mock_config
        test_name = get_test_name()
        self.assertEqual(test_name, 'gpt-35-turbo_hybrid_recursive_900_50_hyde')

if __name__ == '__main__':
    unittest.main()