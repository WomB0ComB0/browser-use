#!/usr/bin/env python3
import os
import subprocess
import sys
import time


def run_dashboard():
    print("Starting Dashboard process...")
    subprocess.run([sys.executable, "run_pipeline.py", "dashboard", "--port", "8081"])

def run_demo():
    # Start Dashboard in a separate thread/process
    # Since subprocess.run blocks, we use Popen or a thread
    dashboard_process = subprocess.Popen(
        [sys.executable, "run_pipeline.py", "dashboard", "--port", "8081"],
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid  # Create new process group
    )
    
    print("\nWaiting 5 seconds for dashboard to start...")
    time.sleep(5)
    print("\nDashboard is running at http://localhost:8081")
    print("Open this URL in your browser to see live logs.\n")
    
    # Create demo input file
    os.makedirs("data", exist_ok=True)
    demo_file = "data/demo_input.txt"
    if not os.path.exists(demo_file):
        with open(demo_file, "w") as f:
            f.write("Topic: The Future of Quantum Computing in Finance")
    
    print("Running Demo Workflow...")
    try:
        # Run the pipeline processing the file with the demo workflow
        subprocess.run([
            sys.executable, "run_pipeline.py", "process", 
            demo_file,
            "--workflow", "demo_parallel"
        ], check=True)
        
        print("\nDemo completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError running demo: {e}")
    finally:
        print("\nStopping dashboard...")
        os.killpg(os.getpgid(dashboard_process.pid), signal.SIGTERM)
        dashboard_process.wait()
        print("Dashboard stopped.")

if __name__ == "__main__":
    import signal
    run_demo()
