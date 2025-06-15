#!/usr/bin/env python3
"""
Monitor benchmark progress in real-time.
"""

import time
import subprocess
import os
import re
from datetime import datetime

def get_benchmark_status():
    """Get current benchmark status."""
    try:
        # Check if process is running
        result = subprocess.run(['pgrep', '-f', 'evaluate.py'], capture_output=True, text=True)
        if not result.stdout.strip():
            return "STOPPED", "Benchmark not running"
        
        # Read log file
        if not os.path.exists('benchmark_full.log'):
            return "STARTING", "Log file not found"
        
        with open('benchmark_full.log', 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return "STARTING", "Log file empty"
        
        # Find current model and progress
        current_model = "Unknown"
        current_progress = "0%"
        time_remaining = "Unknown"
        
        # Look for model info
        for line in reversed(lines[-50:]):  # Check last 50 lines
            if "Running evaluation with" in line:
                current_model = line.split("Running evaluation with")[-1].strip()
                break
        
        # Look for latest progress
        for line in reversed(lines[-20:]):  # Check last 20 lines
            # Match progress pattern like "Processing sec14-1 (qwen2.5vl-7b):  70%|███████   | 14/20"
            match = re.search(r'Processing .+\([\w\-\.]+\):\s+(\d+)%.*\|\s+(\d+)/(\d+).*\[.*<(.+?),', line)
            if match:
                percent = match.group(1)
                current = match.group(2)
                total = match.group(3)
                time_rem = match.group(4)
                current_progress = f"{percent}% ({current}/{total})"
                time_remaining = time_rem
                break
        
        return "RUNNING", f"Model: {current_model} | Progress: {current_progress} | ETA: {time_remaining}"
        
    except Exception as e:
        return "ERROR", str(e)

def display_progress():
    """Display progress with nice formatting."""
    print("\n" + "="*80)
    print("🔥 SAI-BENCHMARK PROGRESS MONITOR")
    print("="*80)
    
    models_order = [
        "Qwen 2.5-VL 7B", "LLaVA-Phi3", "MiniCPM-V 2.6", 
        "BakLLaVA", "Gemma 3 27B Vision", "LLaMA 3.2 Vision 11B"
    ]
    
    while True:
        try:
            status, message = get_benchmark_status()
            current_time = datetime.now().strftime("%H:%M:%S")
            
            print(f"\r[{current_time}] {status}: {message}", end="", flush=True)
            
            if status == "STOPPED":
                print("\n\n✅ Benchmark completed! Checking results...")
                
                # Look for results file
                import glob
                result_files = glob.glob("out/multi_model_evaluation_*.json")
                if result_files:
                    latest_result = max(result_files, key=os.path.getctime)
                    print(f"📊 Results saved to: {latest_result}")
                    
                    # Try to show summary
                    try:
                        import json
                        with open(latest_result, 'r') as f:
                            data = json.load(f)
                        
                        print("\n🏆 QUICK SUMMARY:")
                        summary = data.get('summary', {})
                        for model_name, stats in summary.items():
                            print(f"  {model_name}: EFS {stats.get('avg_early_fire_score', 0):.3f}, "
                                  f"Acc {stats.get('accuracy', 0):.3f}")
                    except:
                        print("📄 Check the results file for detailed metrics.")
                
                break
            elif status == "ERROR":
                print(f"\n❌ Error: {message}")
                break
            
            time.sleep(5)  # Update every 5 seconds
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Monitor stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Monitor error: {e}")
            break

if __name__ == "__main__":
    display_progress()