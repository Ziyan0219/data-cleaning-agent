"""
Test script for the cattle data cleaning system.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the fixed controller directly
from src.agents.main_controller_fixed import process_cleaning_request

async def test_cattle_cleaning():
    """Test the cattle data cleaning system."""
    try:
        print("🧪 Testing Cattle Data Cleaning System")
        print("=" * 50)
        
        # Test data path
        test_data_path = "data/input/cattle_data.csv"
        
        if not Path(test_data_path).exists():
            print(f"❌ Test data not found: {test_data_path}")
            return False
        
        print(f"📁 Using test data: {test_data_path}")
        
        # Test requirements
        requirements = "Process cattle data for weight validation and label correction"
        print(f"📋 Requirements: {requirements}")
        print()
        
        # Process the data
        print("🔄 Processing data...")
        result = await process_cleaning_request(
            user_requirements=requirements,
            data_source=test_data_path
        )
        
        # Check results
        if result.get("status") == "completed":
            print("✅ Processing completed successfully!")
            print(f"⏱️ Execution time: {result.get('execution_time', 0):.2f} seconds")
            print(f"📊 Quality score: {result.get('quality_score', 0):.1f}%")
            print(f"📈 Original rows: {result.get('original_rows', 0)}")
            print(f"📉 Cleaned rows: {result.get('cleaned_rows', 0)}")
            print()
            print("📋 REPORT:")
            print("-" * 30)
            print(result.get("final_report", "No report generated"))
            
            # Save test results
            if result.get("cleaned_data"):
                output_path = "data/output/test_cleaned_data.csv"
                os.makedirs("data/output", exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(result["cleaned_data"])
                print(f"\n💾 Cleaned data saved to: {output_path}")
            
            return True
        else:
            print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_cattle_cleaning())
    if success:
        print("\n🎉 All tests passed! System is working correctly.")
        sys.exit(0)
    else:
        print("\n💥 Tests failed! Please check the errors above.")
        sys.exit(1)

