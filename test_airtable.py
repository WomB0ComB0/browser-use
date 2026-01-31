
import asyncio
import sys
from pathlib import Path

import aiofiles
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.config import PipelineConfig
from pipeline.extractors import get_extractor_for_file
from pipeline.processor import PipelineProcessor
from pipeline.utils.browser_executor import BrowserExecutor


async def main():
    config = PipelineConfig.load()
    processor = PipelineProcessor(config)
    processor.initialize()
    
    data_dir = Path("data/resq")
    url = "https://airtable.com/appxDXHfPCZvb75qk/pagyqQLVvYMoPT9pg/form"
    
    print(f"Reading files from {data_dir}...")
    aggregated_content = ""
    
    # Iterate over all files in the directory
    # Iterate over all files in the directory
    # Prioritize information.txt
    files = sorted(list(data_dir.iterdir()), key=lambda x: (0 if x.name == 'information.txt' else 1, x.name))
    
    for file_path in files:
        if file_path.is_file():
            print(f"Processing {file_path.name}...")
            try:
                extractor = get_extractor_for_file(file_path)
                result = extractor.extract(file_path)
                aggregated_content += f"\n\n--- Source: {file_path.name} ---\n{result.content}"
            except Exception as e:
                print(f"Error reading {file_path.name}: {e}")

    # Create a dummy ExtractedContent with the aggregated text
    # We import ExtractedContent to do this properly
    from datetime import datetime

    from pipeline.extractors.base import ExtractedContent
    
    content = ExtractedContent(
        content=aggregated_content,
        summary="Aggregated content from data/resq",
        file_path=data_dir,
        file_type="Aggregated",
        file_size_bytes=len(aggregated_content),
        modified_time=datetime.now()
    )
    
    # Load custom workflow
    workflow_path = Path("pipeline/workflows/airtable_extraction.yaml")
    async with aiofiles.open(workflow_path, mode='r') as f:
        workflow_content = await f.read()
        workflow = yaml.safe_load(workflow_content)
        
    print("Running extraction workflow...")
    result = await processor.orchestrator.execute_workflow(workflow, content)
    
    if not result.success:
        print("Workflow failed!")
        return
        
    print(f"Extraction result: {result.final_output}")
    
    # Parse JSON from result (assuming one block)
    data = processor.orchestrator.extract_json_from_output(result.final_output)
    if not data:
        pass
        
    print(f"Structured Data: {data}")
    
    print("Starting Browser Agent...")
    executor = BrowserExecutor(config)
    try:
        browser_result = await executor.fill_form(url, data)
        print("Browser Agent Result:", browser_result)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
