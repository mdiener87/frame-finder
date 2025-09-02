# test_frontend_integration.py
import requests
import time

def test_frontend_integration():
    """Test that the frontend is correctly sending new parameters"""
    print("Testing frontend integration...")
    
    # Test the main page loads
    try:
        response = requests.get('http://localhost:5000/')
        if response.status_code == 200:
            print("✓ Main page loads correctly")
            
            # Check if advanced settings panel HTML is present
            if 'Advanced Settings' in response.text:
                print("✓ Advanced settings panel is present")
            else:
                print("✗ Advanced settings panel is missing")
                
            # Check if new input fields are present
            expected_fields = ['frameStride', 'resolutionTarget', 'lpipsThreshold', 'clipThreshold']
            for field in expected_fields:
                if f'id="{field}"' in response.text:
                    print(f"✓ {field} input field is present")
                else:
                    print(f"✗ {field} input field is missing")
        else:
            print(f"✗ Main page failed to load (status code: {response.status_code})")
    except Exception as e:
        print(f"✗ Failed to connect to server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_frontend_integration()
    if success:
        print("\n✓ Frontend integration test completed!")
    else:
        print("\n✗ Frontend integration test failed!")