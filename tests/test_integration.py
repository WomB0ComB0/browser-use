import asyncio
import shutil
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# sys.modules mocks removed to allow real imports in verified environment
from pipeline.config import PipelineConfig
from pipeline.extractors.ocr_extractor import OCRExtractor
from pipeline.generators.base import GeneratedInstructions
from pipeline.memory.pinecone_service import PineconeMemory
from pipeline.processor import PipelineProcessor
from pipeline.watcher import DebouncedHandler


class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = Path("data/test_integration")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.config = PipelineConfig()
        self.config.watcher.debounce_seconds = 0.1

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_watcher_race_condition(self):
        """Test that rapid creation and deletion doesn't crash the watcher."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        callback = MagicMock()
        handler = DebouncedHandler(callback, self.config, loop)
        
        test_file = self.test_dir / "race.pdf"
        
        # Simulate create event
        event = MagicMock()
        event.is_directory = False
        event.src_path = str(test_file)
        
        # Create file
        test_file.touch()
        handler.on_created(event)
        
        # Delete file immediately
        if test_file.exists():
            test_file.unlink()
            
        # Run loop for a bit to let debounce expire
        loop.run_until_complete(asyncio.sleep(0.2))
        
        # Callback should NOT have been called because file is gone
        callback.assert_not_called()
        
        loop.close()

    @patch("pipeline.extractors.ocr_extractor.convert_from_path")
    @patch("pipeline.extractors.ocr_extractor.easyocr.Reader")
    def test_ocr_extractor(self, mock_reader_class, mock_convert):
        """Test OCR extraction logic."""
        # Mock dependencies
        mock_convert.return_value = ["mock_image_object"]
        
        mock_reader = MagicMock()
        # readtext with detail=0 returns a list of strings
        mock_reader.readtext.return_value = ["Extracted", "Text", "from", "Image"]
        mock_reader_class.return_value = mock_reader
        
        extractor = OCRExtractor()
        test_file = self.test_dir / "scan.pdf"
        test_file.touch()
        
        result = extractor.extract(test_file)
        
        self.assertIn("Extracted\nText\nfrom\nImage", result.content)
        self.assertIn("OCR", result.file_type)

    @patch("pipeline.memory.pinecone_service.Pinecone")
    @patch("pipeline.memory.pinecone_service.SentenceTransformer")
    def test_pinecone_memory(self, mock_transformer, mock_pinecone):
        """Test Pinecone memory service."""
        # Mock Pinecone
        mock_index = MagicMock()
        mock_client = MagicMock()
        mock_client.Index.return_value = mock_index
        mock_client.list_indexes.return_value = [MagicMock(name="browser-use-memory")]
        mock_pinecone.return_value = mock_client
        
        # Mock SentenceTransformer
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2, 0.3])
        mock_transformer.return_value = mock_model
        
        # Config with API key
        self.config.memory.pinecone_api_key = "test-key"
        self.config.memory.pinecone_index_name = "browser-use-memory"
        
        memory = PineconeMemory(self.config)
        
        # Test Upsert
        success = memory.upsert("Test content", "test.pdf")
        self.assertTrue(success)
        mock_index.upsert.assert_called_once()
        
        # Test Query
        mock_match = MagicMock()
        mock_match.id = "doc1"
        mock_match.metadata = {"text": "Result content"}
        mock_match.score = 0.9
        mock_index.query.return_value = MagicMock(matches=[mock_match])
        
        results = memory.query("query")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["content"], "Result content")

    @patch("pipeline.processor.get_extractor_for_file")
    @patch("pipeline.processor.get_generator")
    @patch("pipeline.processor.PineconeMemory")
    def test_processor_memory_integration(self, mock_memory_class, mock_get_generator, mock_get_extractor):
        """Test that PipelineProcessor uses memory when enabled."""
        # 1. Setup Logic
        # Mock Memory
        mock_memory_instance = MagicMock()
        mock_memory_instance.enabled = True
        mock_memory_class.return_value = mock_memory_instance
        
        # 2. Execution
        # We need to run async process_file
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Mock Generator
        mock_generator = MagicMock()
        mock_generated_instructions = GeneratedInstructions(
            instructions="Output markdown",
            title="Title",
            source_file=Path("test.txt"),
            source_type="txt",
            model_used="test-model"
        )
        # Create future attached to the loop we just set
        future = loop.create_future()
        future.set_result(mock_generated_instructions)
        mock_generator.generate.return_value = future
        mock_get_generator.return_value = mock_generator

        processor = PipelineProcessor(self.config)
        processor.generator = mock_generator # Manually set since we skip initialize()
        processor.memory = mock_memory_instance # Manually set memory since we skip initialize()
        # Manually set memory instance if __init__ mocking is tricky, 
        # but mock_memory_class should handle the instantiation in __init__
        
        test_file = Path("test.txt")
        # Touch file so .stat() works
        with open("test.txt", "w") as f:
            f.write("content")
            
        try:
            loop.run_until_complete(processor.process_file(test_file))
        finally:
            Path("test.txt").unlink(missing_ok=True)
            loop.close()

        # 3. Verification
        # Check that upsert was called
        mock_memory_instance.upsert.assert_called_once()
        call_args = mock_memory_instance.upsert.call_args
        self.assertEqual(call_args.kwargs["content"], "Output markdown")
        self.assertEqual(call_args.kwargs["source_file"], "test.txt")


if __name__ == "__main__":
    unittest.main()
