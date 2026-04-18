# This thing is for testing if the pipeline from interface.py works in returning results
import os
import sys
from pathlib import Path

# Add the project root to the python path so we can import from src
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.interface import get_match

def test_single_image(image_path: str):
    """
    Tests the get_match function with a single image path.
    """
    print(f"--- Testing Image: {image_path} ---")
    
    if not os.path.exists(image_path):
        print(f"Error: File not found at '{image_path}'")
        return
        
    try:
        # Run inference
        result = get_match(image_path, k=5)
        print(result)
        
        # Display results
        if isinstance(result, dict) and "error" in result: # if the sim search fails it will return return {"error": "No matching records found in the database."}
            print("Oops:", result["error"])
        else: 
            for i, match in enumerate(result):
                print(f"\n--- Match {i + 1} ---")
                print(f"You look like: {match.get('Name', 'Unknown')}")
                print(f"Similarity Distance: {match.get('Similarity Score', 'N/A')}")
                print(f"Bio: {match.get('Bio', 'N/A')}")
                print(f"Matched Image: {match.get('Image Path', 'N/A')}")
            
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")

if __name__ == "__main__":
    # Point this to an actual image inside your data/samples folder
    # For example, if you place a selfie named 'test_face.jpg' inside data/samples/
    sample_img = os.path.join(project_root, "data", "samples", "k4.jpg")
    
    test_single_image(str(sample_img))
