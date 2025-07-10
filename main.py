"""
Data Cleaning Agent System Main Program

This is the main entry point for the Data Cleaning Agent system,
providing a command-line interface for executing data cleaning tasks.
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agents.main_controller import process_cleaning_request
from src.config.settings import get_settings


async def main_cleaning_task(requirements: str, data_source: str, 
                           output_file: Optional[str] = None) -> bool:
    """Execute main cleaning task"""
    try:
        print("🚀 Starting data cleaning task")
        print(f"Data source: {data_source}")
        print(f"Cleaning requirements: {requirements}")
        
        # Execute cleaning
        result = await process_cleaning_request(
            user_requirements=requirements,
            data_source=data_source
        )
        
        # Process results
        if result['status'] == 'completed':
            print("✅ Data cleaning completed")
            print(f"Session ID: {result['session_id']}")
            print(f"Execution time: {result.get('execution_time', 0):.2f} seconds")
            
            # Display quality metrics
            quality_metrics = result.get('quality_metrics', {})
            if quality_metrics:
                print("📊 Quality Metrics:")
                for metric, score in quality_metrics.items():
                    print(f"  {metric}: {score:.2f}")
            
            # Save results
            if output_file and result.get('final_data'):
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result['final_data'])
                
                print(f"💾 Cleaning results saved to: {output_path}")
            
            return True
            
        else:
            print(f"❌ Data cleaning failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        return False


def test_configuration() -> bool:
    """Test system configuration"""
    print("🔧 Testing system configuration")
    
    settings = get_settings()
    
    # Validate configuration
    if not settings.validate_config():
        print("❌ Configuration validation failed")
        return False
    
    # Test LLM connection
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            model=settings.llm.model,
            api_key=settings.llm.api_key,
            temperature=0,
            timeout=10
        )
        
        response = llm.invoke("Hello, this is a test message.")
        print("✅ LLM connection test successful")
        return True
        
    except Exception as e:
        print(f"❌ LLM connection test failed: {str(e)}")
        return False


def show_examples():
    """Display usage examples"""
    examples = """
🤖 Data Cleaning Agent System Usage Examples

Basic usage:
  python main.py --requirements "Handle missing values and duplicates" --data-source "data/input/customers.csv"

Specify output file:
  python main.py --requirements "Clean sales data" --data-source "sales.xlsx" --output "cleaned_sales.csv"

Complex cleaning requirements:
  python main.py --requirements "
    1. Handle all missing values, use mean fill for numeric columns
    2. Remove duplicate records, keep the latest ones
    3. Standardize email and phone number formats
    4. Detect and flag unusual sales amounts
  " --data-source "complex_data.csv"

Test configuration:
  python main.py --test-config

Show help:
  python main.py --help
"""
    print(examples)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Data Cleaning Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --requirements "Handle missing values" --data-source "data.csv"
  python main.py --test-config
  python main.py --examples
        """
    )
    
    # Main arguments
    parser.add_argument(
        '--requirements', '-r',
        type=str,
        help='Data cleaning requirements description'
    )
    
    parser.add_argument(
        '--data-source', '-d',
        type=str,
        help='Data source file path or data content'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path (optional)'
    )
    
    # Function arguments
    parser.add_argument(
        '--test-config',
        action='store_true',
        help='Test system configuration'
    )
    
    parser.add_argument(
        '--examples',
        action='store_true',
        help='Show usage examples'
    )
    
    # Configuration arguments
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Log level'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Specify LLM model to use'
    )
    
    return parser


async def async_main():
    """Async main function"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set log level
    import os
    os.environ['LOG_LEVEL'] = args.log_level
    
    # Set model (if specified)
    if args.model:
        os.environ['OPENAI_MODEL'] = args.model
    
    # Handle different commands
    if args.test_config:
        success = test_configuration()
        sys.exit(0 if success else 1)
    
    elif args.examples:
        show_examples()
        sys.exit(0)
    
    elif args.requirements and args.data_source:
        # Execute cleaning task
        success = await main_cleaning_task(
            requirements=args.requirements,
            data_source=args.data_source,
            output_file=args.output
        )
        sys.exit(0 if success else 1)
    
    else:
        # Show help information
        parser.print_help()
        print("\n💡 Tip: Use --examples to see detailed usage examples")
        sys.exit(1)


def main():
    """Main function entry point"""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("👋 User interrupted, program exiting")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Program exited with exception: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

