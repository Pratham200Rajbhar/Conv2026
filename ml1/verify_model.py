import subprocess
import os

def run_verify(script_path):
    print(f"\n{'='*50}")
    print(f"Running verification for: {script_path}")
    print(f"{'='*50}\n")
    try:
        result = subprocess.run(["python3", script_path], capture_output=False, text=True)
        if result.returncode == 0:
            print(f"\n✅ {script_path} verified successfully!")
        else:
            print(f"\n❌ {script_path} verification failed!")
    except Exception as e:
        print(f"\n⚠️ Error running {script_path}: {e}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    scripts = [
        os.path.join(BASE_DIR, "crop_yield_prediction", "verify_model.py"),
        os.path.join(BASE_DIR, "plant_disease_classification", "verify_model.py")
    ]
    
    for script in scripts:
        if os.path.exists(script):
            run_verify(script)
        else:
            print(f"Error: Script not found at {script}")
