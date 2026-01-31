import asyncio
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from pipeline.extractors.base import ExtractedContent
from pipeline.generators.ollama import OllamaGenerator


class TestOllamaGenerator(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.config.generator = MagicMock()
        self.config.generator.max_tokens = 100
        self.config.generator.temperature = 0.7
        self.config.generator.instruction_template = "Type: {file_type}, Summary: {summary}"
        self.config.generator.host = "http://localhost:11434"

    @patch('pipeline.generators.ollama.OllamaAsyncClient')
    def test_generate_llama3_options(self, mock_client_cls):
        # Setup
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.return_value = {'message': {'content': 'Generated instructions'}}
        
        self.config.generator.model = "llama3:latest"
        generator = OllamaGenerator(self.config)
        
        content = ExtractedContent(
            content="some text",
            summary="A summary",
            file_type="PDF",
            file_path=Path("test.pdf"),
            file_size_bytes=100,
            modified_time=datetime.now(),
            structure={}
        )

        # Execute
        asyncio.run(generator.generate(content))

        # Verify
        mock_client.chat.assert_called_once()
        call_kwargs = mock_client.chat.call_args.kwargs
        options = call_kwargs['options']
        
        self.assertIn('num_ctx', options)
        self.assertEqual(options['num_ctx'], 8192)
        self.assertIn('stop', options)
        self.assertIn('<|eot_id|>', options['stop'])
        self.assertEqual(call_kwargs['model'], "llama3:latest")

    @patch('pipeline.generators.ollama.OllamaAsyncClient')
    def test_generate_other_model_options(self, mock_client_cls):
        # Setup
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.return_value = {'message': {'content': 'Generated instructions'}}
        
        self.config.generator.model = "mistral"
        generator = OllamaGenerator(self.config)
        
        content = ExtractedContent(
            content="some text",
            summary="A summary",
            file_type="PDF",
            file_path=Path("test.pdf"),
            file_size_bytes=100,
            modified_time=datetime.now(),
            structure={}
        )

        # Execute
        asyncio.run(generator.generate(content))

        # Verify
        mock_client.chat.assert_called_once()
        call_kwargs = mock_client.chat.call_args.kwargs
        options = call_kwargs['options']
        
        self.assertNotIn('num_ctx', options)
        self.assertNotIn('stop', options)
        self.assertEqual(call_kwargs['model'], "mistral")

if __name__ == '__main__':
    unittest.main()
