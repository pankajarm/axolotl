#!/usr/bin/env python3
"""
Training Monitor for Devstral-Small-2-24B-Instruct-2512 on H100
Monitors training progress, GPU stats, and estimates completion time.

Usage:
    python monitor_devstral2_h100.py [--log-file LOG_FILE] [--interval SECONDS]
"""

import argparse
import glob
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

# ANSI colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# Using Devstral-Small-2-24B-Instruct-2512 (FP8) on H100
LOG_DIR = Path("/home/ubuntu/us-south-2-nano-chat-exp/logs")
OUTPUT_DIR = "/home/ubuntu/us-south-2-nano-chat-exp/outputs/Devstral-Small-2-24B-Instruct-2512-sft-v1/lora-out"


def get_latest_log_file() -> str:
    """Get the most recent training log file."""
    log_files = glob.glob(str(LOG_DIR / "devstral2_h100_full_*.log"))
    if log_files:
        return max(log_files, key=lambda x: Path(x).stat().st_mtime)
    return str(LOG_DIR / "devstral2_h100_full_latest.log")


def get_gpu_stats():
    """Get GPU memory and utilization stats."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,name", 
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 5:
                mem_used = float(parts[0]) / 1024  # Convert to GB
                mem_total = float(parts[1]) / 1024
                gpu_util = int(parts[2])
                gpu_temp = int(parts[3])
                gpu_name = parts[4]
                return {
                    "mem_used": mem_used,
                    "mem_total": mem_total,
                    "mem_pct": (mem_used / mem_total) * 100,
                    "gpu_util": gpu_util,
                    "gpu_temp": gpu_temp,
                    "gpu_name": gpu_name
                }
    except Exception:
        pass
    return None


def parse_training_log(log_file):
    """Parse the training log to extract progress information."""
    if not Path(log_file).exists():
        return None
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
    except Exception:
        return None
    
    progress = {
        "current_step": 0,
        "max_steps": 0,
        "current_epoch": 0.0,
        "max_epochs": 3,
        "loss": None,
        "learning_rate": None,
        "tokens_per_second": None,
        "start_time": None,
        "last_update": None
    }
    
    # Look for max_steps in config output
    max_steps_match = re.search(r'"max_steps":\s*(\d+)', content)
    if max_steps_match:
        progress["max_steps"] = int(max_steps_match.group(1))
    
    # Look for num_epochs
    epochs_match = re.search(r'"num_train_epochs":\s*(\d+)', content)
    if epochs_match:
        progress["max_epochs"] = int(epochs_match.group(1))
    
    # Find training start time
    start_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', content)
    if start_match:
        try:
            progress["start_time"] = datetime.strptime(start_match.group(1), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    
    # Parse training log lines for step progress
    lines = content.split('\n')
    
    for line in reversed(lines[-500:]):  # Check last 500 lines
        step_match = re.search(r"'step':\s*(\d+)", line)
        loss_match = re.search(r"'loss':\s*([\d.]+)", line)
        epoch_match = re.search(r"'epoch':\s*([\d.]+)", line)
        lr_match = re.search(r"'learning_rate':\s*([\d.e-]+)", line)
        tps_match = re.search(r"'tokens_per_second_per_gpu':\s*([\d.]+)", line)
        
        if step_match and progress["current_step"] == 0:
            progress["current_step"] = int(step_match.group(1))
        if loss_match and progress["loss"] is None:
            progress["loss"] = float(loss_match.group(1))
        if epoch_match and progress["current_epoch"] == 0:
            progress["current_epoch"] = float(epoch_match.group(1))
        if lr_match and progress["learning_rate"] is None:
            progress["learning_rate"] = float(lr_match.group(1))
        if tps_match and progress["tokens_per_second"] is None:
            progress["tokens_per_second"] = float(tps_match.group(1))
        
        # Check for percentage progress
        pct_match = re.search(r'(\d+)%\|', line)
        if pct_match and progress["current_step"] == 0:
            step_in_bar = re.search(r'\|\s*(\d+)/(\d+)', line)
            if step_in_bar:
                progress["current_step"] = int(step_in_bar.group(1))
                if progress["max_steps"] == 0:
                    progress["max_steps"] = int(step_in_bar.group(2))
    
    # Check for completion
    if "Training completed" in content or "train completed" in content.lower():
        progress["completed"] = True
    else:
        progress["completed"] = False
    
    return progress


def check_screen_running():
    """Check if training screen is running."""
    try:
        result = subprocess.run(
            ["screen", "-ls"], capture_output=True, text=True, timeout=5
        )
        return "devstral2_h100" in result.stdout
    except Exception:
        return False


def check_checkpoints():
    """Check for saved checkpoints."""
    output_path = Path(OUTPUT_DIR)
    checkpoints = list(output_path.glob("checkpoint-*"))
    return sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0)


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 0:
        return "N/A"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_status(log_file, clear=True):
    """Print the current training status."""
    if clear:
        print("\033[H\033[J", end="")  # Clear screen
    
    print(f"{Colors.BOLD}{Colors.CYAN}=" * 70)
    print(f"  🚀 Devstral-Small-2-24B-Instruct-2512 Training Monitor (H100)")
    print(f"=" * 70 + f"{Colors.ENDC}")
    print(f"{Colors.YELLOW}Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print()
    
    # Check screen status
    screen_running = check_screen_running()
    if screen_running:
        print(f"{Colors.GREEN}✓ Training screen is running{Colors.ENDC}")
    else:
        print(f"{Colors.RED}✗ Training screen not found{Colors.ENDC}")
    print()
    
    # GPU Stats
    gpu = get_gpu_stats()
    if gpu:
        mem_color = Colors.GREEN if gpu["mem_pct"] < 85 else (Colors.YELLOW if gpu["mem_pct"] < 95 else Colors.RED)
        util_color = Colors.GREEN if gpu["gpu_util"] > 80 else (Colors.YELLOW if gpu["gpu_util"] > 50 else Colors.RED)
        
        print(f"{Colors.BOLD}GPU Status ({gpu['gpu_name']}):{Colors.ENDC}")
        print(f"  Memory: {mem_color}{gpu['mem_used']:.1f}/{gpu['mem_total']:.1f} GB ({gpu['mem_pct']:.1f}%){Colors.ENDC}")
        print(f"  Utilization: {util_color}{gpu['gpu_util']}%{Colors.ENDC}")
        print(f"  Temperature: {gpu['gpu_temp']}°C")
        print()
    
    # Training Progress
    progress = parse_training_log(log_file)
    if progress:
        print(f"{Colors.BOLD}Training Progress:{Colors.ENDC}")
        
        if progress["max_steps"] > 0:
            pct = (progress["current_step"] / progress["max_steps"]) * 100
            bar_width = 40
            filled = int(bar_width * pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"  Step: {Colors.CYAN}{progress['current_step']}/{progress['max_steps']}{Colors.ENDC}")
            print(f"  [{Colors.GREEN}{bar}{Colors.ENDC}] {pct:.1f}%")
        else:
            print(f"  Step: {Colors.CYAN}{progress['current_step']}{Colors.ENDC}")
        
        print(f"  Epoch: {Colors.CYAN}{progress['current_epoch']:.2f}/{progress['max_epochs']}{Colors.ENDC}")
        
        if progress["loss"] is not None:
            loss_color = Colors.GREEN if progress["loss"] < 1.0 else (Colors.YELLOW if progress["loss"] < 2.0 else Colors.RED)
            print(f"  Loss: {loss_color}{progress['loss']:.4f}{Colors.ENDC}")
        
        if progress["learning_rate"] is not None:
            print(f"  Learning Rate: {progress['learning_rate']:.2e}")
        
        if progress["tokens_per_second"] is not None:
            print(f"  Tokens/sec: {Colors.CYAN}{progress['tokens_per_second']:.1f}{Colors.ENDC}")
        
        # Estimate time remaining
        if progress["max_steps"] > 0 and progress["current_step"] > 0 and progress["start_time"]:
            elapsed = (datetime.now() - progress["start_time"]).total_seconds()
            steps_remaining = progress["max_steps"] - progress["current_step"]
            time_per_step = elapsed / progress["current_step"]
            eta_seconds = steps_remaining * time_per_step
            
            print()
            print(f"  Elapsed: {format_time(elapsed)}")
            print(f"  ETA: {Colors.CYAN}{format_time(eta_seconds)}{Colors.ENDC}")
        
        if progress.get("completed"):
            print()
            print(f"{Colors.GREEN}{Colors.BOLD}✓ Training Completed!{Colors.ENDC}")
        
        print()
    else:
        print(f"{Colors.YELLOW}Waiting for training to start...{Colors.ENDC}")
        print(f"Log file: {log_file}")
        if not Path(log_file).exists():
            print(f"{Colors.RED}Log file not found yet{Colors.ENDC}")
        print()
    
    # Checkpoints
    checkpoints = check_checkpoints()
    if checkpoints:
        print(f"{Colors.BOLD}Saved Checkpoints:{Colors.ENDC}")
        for cp in checkpoints[-4:]:  # Show last 4
            print(f"  📁 {cp.name}")
        if len(checkpoints) > 4:
            print(f"  ... and {len(checkpoints) - 4} more")
        print()
    
    print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}")
    print(f"Press Ctrl+C to exit | Log: {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Monitor Devstral-2 H100 training progress")
    parser.add_argument("--log-file", "-l", default=None, help="Path to training log file")
    parser.add_argument("--interval", "-i", type=int, default=10, help="Refresh interval in seconds")
    parser.add_argument("--once", "-o", action="store_true", help="Print status once and exit")
    args = parser.parse_args()
    
    log_file = args.log_file or get_latest_log_file()
    
    if args.once:
        print_status(log_file, clear=False)
        return
    
    print("Starting training monitor... (Press Ctrl+C to exit)")
    try:
        while True:
            print_status(log_file)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()

