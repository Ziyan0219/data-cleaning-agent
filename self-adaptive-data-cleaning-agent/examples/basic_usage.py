"""
Basic Usage Example

This example demonstrates how to use the Data Cleaning Agent system
for basic data cleaning operations.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.main_controller import process_cleaning_request
from src.config.settings import get_settings


async def basic_cleaning_example():
    """Basic data cleaning example"""
    print("🚀 Starting Basic Data Cleaning Example")
    print("=" * 50)
    
    # Check configuration
    settings = get_settings()
    if not settings.validate_config():
        print("❌ Configuration validation failed, please check environment variables")
        return
    
    # Sample data (CSV format)
    sample_data = """name,age,email,phone,city
John Doe,25,john@example.com,123-456-7890,New York
Jane Smith,,jane.smith@email.com,987-654-3210,Los Angeles
Bob Johnson,35,bob@invalid,555-0123,Chicago
Alice Brown,28,alice@example.com,555-0124,Houston
John Doe,25,john@example.com,123-456-7890,New York
Mike Wilson,45,mike@example.com,,Miami
Sarah Davis,32,sarah@example.com,555-0125,Seattle"""
    
    # Save sample data to temporary file
    temp_file = project_root / "data" / "temp" / "sample_data.csv"
    temp_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(sample_data)
    
    print(f"📁 Sample data saved to: {temp_file}")
    
    try:
        # Execute data cleaning
        result = await process_cleaning_request(
            user_requirements="""
            Please clean this customer dataset with the following requirements:
            1. Handle missing age and phone number values
            2. Remove duplicate records
            3. Validate and fix email address formats
            4. Standardize phone number formats
            5. Ensure city name consistency
            """,
            data_source=str(temp_file)
        )
        
        # Display results
        print("\n📊 Cleaning Results:")
        print(f"Status: {result['status']}")
        print(f"Session ID: {result['session_id']}")
        
        if result['status'] == 'completed':
            print(f"Execution Time: {result.get('execution_time', 0):.2f} seconds")
            
            # Display quality metrics
            quality_metrics = result.get('quality_metrics', {})
            if quality_metrics:
                print("\n📈 Quality Metrics:")
                for metric, score in quality_metrics.items():
                    print(f"  {metric}: {score:.2f}")
            
            # Display cleaning results summary
            agent_results = result.get('results', {})
            if 'analysis' in agent_results:
                analysis = agent_results['analysis']
                print(f"\n🔍 Analysis Results:")
                print(f"  Issues Found: {analysis.get('analysis_summary', {}).get('total_issues', 0)}")
                print(f"  High Severity Issues: {analysis.get('analysis_summary', {}).get('high_severity_issues', 0)}")
                print(f"  Quality Score: {analysis.get('quality_score', 0):.1f}/100")
            
            # Save cleaned data
            if result.get('final_data'):
                output_file = project_root / "data" / "output" / "cleaned_sample_data.csv"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result['final_data'])
                
                print(f"\n💾 Cleaned data saved to: {output_file}")
        
        elif result['status'] == 'failed':
            print(f"❌ Cleaning failed: {result.get('error', 'Unknown error')}")
            
            # Display partial results (if any)
            partial_results = result.get('partial_results', {})
            if partial_results:
                print("📋 Partial Results:")
                for key, value in partial_results.items():
                    print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        print("Please check configuration and network connection")
    
    finally:
        # Clean up temporary files
        if temp_file.exists():
            temp_file.unlink()
            print(f"\n🧹 Cleaned up temporary file: {temp_file}")


async def test_configuration():
    """Test system configuration"""
    print("🔧 Testing System Configuration")
    print("=" * 30)
    
    settings = get_settings()
    
    # Check API key
    if settings.llm.api_key:
        print("✅ OpenAI API key is configured")
    else:
        print("❌ OpenAI API key not found")
        print("Please set OPENAI_API_KEY in .env file")
        return False
    
    # Check directories
    required_dirs = [
        settings.data.temp_dir,
        settings.data.output_dir,
        "logs"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ Directory exists: {dir_path}")
        else:
            print(f"⚠️  Directory doesn't exist, will create: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
    
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


def show_help():
    """Display help information"""
    print("""
🤖 Data Cleaning Agent System - Basic Usage Example

Usage:
  python examples/basic_usage.py [options]

Options:
  --test-config    Test system configuration
  --help          Show this help message

Examples:
  python examples/basic_usage.py                # Run basic cleaning example
  python examples/basic_usage.py --test-config  # Test configuration

Notes:
1. Make sure all dependencies are installed: pip install -r requirements.txt
2. Configure environment variables: copy .env.template to .env and fill in API key
3. Recommend running configuration test first

For more information, please refer to project documentation.
""")


async def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test-config":
            success = await test_configuration()
            if success:
                print("\n🎉 Configuration test passed, system is ready!")
            else:
                print("\n❌ Configuration test failed, please check settings and try again")
            return
        elif sys.argv[1] == "--help":
            show_help()
            return
    
    # Run basic example
    await basic_cleaning_example()
    
    print("\n🎉 Basic example execution completed!")
    print("💡 Tip: You can modify sample data and cleaning requirements to test different scenarios")


if __name__ == "__main__":
    asyncio.run(main())

