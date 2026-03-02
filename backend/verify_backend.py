import requests
import json
import time
import subprocess
import sys

def verify():
    # Start backend
    print("Starting backend...")
    d = "e:/Sem 6/ML/ML_Project_cardio"
    # Assuming uvicorn is installed and accessible via 'py -m uvicorn' or just 'uvicorn'
    # using Popen to start it in background
    process = subprocess.Popen(["py", "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"], cwd=d, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for startup
    time.sleep(5)
    
    url = "http://127.0.0.1:8000/predict"
    data = {
        "age": 20000,
        "gender": 1,
        "height": 165,
        "weight": 65,
        "ap_hi": 120,
        "ap_lo": 80,
        "cholesterol": 1,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1
    }
    
    try:
        print(f"Sending request to {url}...")
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("Verification PASSED")
        else:
            print("Verification FAILED")
            
    except Exception as e:
        print(f"Error: {e}")
        # Print stdout/stderr from process if it failed
        outs, errs = process.communicate(timeout=5)
        print("Backend Output:", outs.decode())
        print("Backend Error:", errs.decode())
    finally:
        process.terminate()

if __name__ == "__main__":
    verify()
