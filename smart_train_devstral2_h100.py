#!/usr/bin/env python3
"""
Smart Training Script for Devstral-Small-2-24B-Instruct-2512 on H100
Runs quicktest first, validates everything works, then runs full training.

This script is specifically for H100 GPUs (compute capability 9.0) which can
run the FP8 quantized model natively.

Features:
- Runs quicktest end-to-end (~5-10 min)
- Validates checkpoint saving, eval, wandb reporting
- Checks GPU VRAM utilization and suggests batch size adjustments
- Only proceeds to full training if quicktest passes
- Outputs learnings and recommendations

Usage:
    python smart_train_devstral2_h100.py [--quicktest-only] [--skip-quicktest]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

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

# Configuration for H100
SCRIPT_DIR = Path("/home/ubuntu/us-south-2-nano-chat-exp")
AXOLOTL_DIR = SCRIPT_DIR / "axolotl"
VENV_PATH = AXOLOTL_DIR / ".venv-devstral2"
LOG_DIR = SCRIPT_DIR / "logs"

# Using Devstral-Small-2-24B-Instruct-2512 (FP8) - works on H100 (compute 9.0)
QUICKTEST_CONFIG = AXOLOTL_DIR / "examples/devstral2/devstral-small-2-24b-lora-quicktest-h100.yaml"
FULL_CONFIG = AXOLOTL_DIR / "examples/devstral2/devstral-small-2-24b-lora-h100.yaml"
QUICKTEST_OUTPUT_DIR = SCRIPT_DIR / "outputs/Devstral-Small-2-24B-Instruct-2512-quicktest/lora-out"
FULL_OUTPUT_DIR = SCRIPT_DIR / "outputs/Devstral-Small-2-24B-Instruct-2512-sft-v1/lora-out"


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}{Colors.ENDC}\n")


def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")


def print_info(text: str):
    print(f"{Colors.BLUE}ℹ {text}{Colors.ENDC}")


def check_gpu_compatibility() -> bool:
    """Check if GPU has sufficient compute capability for FP8."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 2:
                compute_cap = float(parts[0])
                gpu_name = parts[1]
                print_info(f"GPU: {gpu_name}")
                print_info(f"Compute Capability: {compute_cap}")
                if compute_cap >= 8.9:
                    print_success("GPU supports FP8 operations")
                    return True
                else:
                    print_error(f"GPU compute capability {compute_cap} < 8.9 required for FP8")
                    return False
    except Exception as e:
        print_error(f"Failed to check GPU: {e}")
    return False


def get_gpu_stats() -> Optional[dict]:
    """Get GPU memory and utilization stats."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 4:
                mem_used = float(parts[0]) / 1024  # Convert to GB
                mem_total = float(parts[1]) / 1024
                return {
                    "mem_used_gb": mem_used,
                    "mem_total_gb": mem_total,
                    "mem_pct": (mem_used / mem_total) * 100,
                    "gpu_util": int(parts[2]),
                    "gpu_temp": int(parts[3])
                }
    except Exception:
        pass
    return None


def run_axolotl_train(config_path: Path, log_file: Path) -> tuple[int, str]:
    """Run axolotl training and return exit code and log content."""
    env = os.environ.copy()
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    env["AXOLOTL_DO_NOT_TRACK"] = "0"
    
    # Build command
    activate_cmd = f"source {VENV_PATH}/bin/activate"
    train_cmd = f"cd {AXOLOTL_DIR} && axolotl train {config_path}"
    full_cmd = f"{activate_cmd} && {train_cmd}"
    
    print_info(f"Running: {train_cmd}")
    print_info(f"Log file: {log_file}")
    print()
    
    # Run training
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            full_cmd,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )
        
        # Stream output to both console and file
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            f.write(line)
            f.flush()
        
        process.wait()
    
    # Read log content
    log_content = log_file.read_text() if log_file.exists() else ""
    return process.returncode, log_content


def analyze_quicktest_results(log_content: str, output_dir: Path) -> dict:
    """Analyze quicktest results and return findings."""
    results = {
        "success": True,
        "errors": [],
        "warnings": [],
        "recommendations": [],
        "metrics": {}
    }
    
    # Check for common errors
    if "CUDA out of memory" in log_content or "OutOfMemoryError" in log_content:
        results["success"] = False
        results["errors"].append("CUDA out of memory - reduce batch size or sequence length")
        results["recommendations"].append("Reduce micro_batch_size to 1 or sequence_len to 2048")
    
    if "ValueError" in log_content or "RuntimeError" in log_content:
        # Extract error message
        error_match = re.search(r'(ValueError|RuntimeError): (.+?)(?:\n|$)', log_content)
        if error_match:
            results["success"] = False
            results["errors"].append(f"{error_match.group(1)}: {error_match.group(2)}")
    
    # Check for FP8-specific issues
    if "compute capability" in log_content.lower() and "8.9" in log_content:
        results["success"] = False
        results["errors"].append("FP8 requires compute capability >= 8.9 - check GPU compatibility")
    
    # Check for checkpoint saving
    checkpoints = list(output_dir.glob("checkpoint-*")) if output_dir.exists() else []
    if checkpoints:
        results["metrics"]["checkpoints_saved"] = len(checkpoints)
        print_success(f"Checkpoint saving works ({len(checkpoints)} checkpoints)")
    else:
        results["warnings"].append("No checkpoints found - checkpoint saving may have failed")
    
    # Check for eval results
    if "eval_loss" in log_content or "'eval_loss'" in log_content:
        eval_match = re.search(r"'eval_loss':\s*([\d.]+)", log_content)
        if eval_match:
            results["metrics"]["eval_loss"] = float(eval_match.group(1))
            print_success(f"Evaluation works (eval_loss: {eval_match.group(1)})")
    else:
        results["warnings"].append("No eval_loss found - evaluation may have failed")
    
    # Check for wandb
    if "wandb:" in log_content.lower() or "Syncing run" in log_content:
        print_success("Weights & Biases logging works")
        results["metrics"]["wandb_enabled"] = True
    else:
        results["warnings"].append("No wandb logging detected")
        results["metrics"]["wandb_enabled"] = False
    
    # Check for training loss
    loss_matches = re.findall(r"'loss':\s*([\d.]+)", log_content)
    if loss_matches:
        losses = [float(l) for l in loss_matches]
        results["metrics"]["initial_loss"] = losses[0]
        results["metrics"]["final_loss"] = losses[-1]
        results["metrics"]["loss_trend"] = "decreasing" if losses[-1] < losses[0] else "not decreasing"
        print_success(f"Training loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    
    # Check GPU memory usage from log
    mem_matches = re.findall(r"memory/max_allocated \(GiB\)':\s*([\d.]+)", log_content)
    if mem_matches:
        max_mem = max(float(m) for m in mem_matches)
        results["metrics"]["max_gpu_memory_gb"] = max_mem
        
        # Get total GPU memory (H100 80GB)
        gpu_stats = get_gpu_stats()
        if gpu_stats:
            total_mem = gpu_stats["mem_total_gb"]
            mem_pct = (max_mem / total_mem) * 100
            results["metrics"]["gpu_memory_pct"] = mem_pct
            
            if mem_pct < 60:
                results["recommendations"].append(
                    f"GPU memory only {mem_pct:.1f}% utilized. H100 has plenty of headroom - "
                    f"consider increasing micro_batch_size to 4 or sequence_len to 16384."
                )
            elif mem_pct < 80:
                results["recommendations"].append(
                    f"GPU memory at {mem_pct:.1f}%. Consider increasing micro_batch_size for better throughput."
                )
            elif mem_pct > 95:
                results["recommendations"].append(
                    f"GPU memory at {mem_pct:.1f}%. May be too aggressive - consider reducing batch size for stability."
                )
            else:
                print_success(f"GPU memory utilization: {mem_pct:.1f}% (optimal range)")
    
    # Check for safetensor saving
    safetensor_files = list(output_dir.glob("**/*.safetensors")) if output_dir.exists() else []
    if safetensor_files:
        print_success(f"Safetensor saving works ({len(safetensor_files)} files)")
        results["metrics"]["safetensors_saved"] = len(safetensor_files)
    
    # Check training completed
    if "Training completed" in log_content or "train completed" in log_content.lower():
        print_success("Training completed successfully")
        results["metrics"]["completed"] = True
    else:
        if results["success"]:  # Only warn if no other errors
            results["warnings"].append("Training may not have completed normally")
    
    return results


def generate_recommendations(results: dict) -> str:
    """Generate recommendations based on quicktest results."""
    lines = []
    
    if results["errors"]:
        lines.append(f"\n{Colors.RED}ERRORS:{Colors.ENDC}")
        for err in results["errors"]:
            lines.append(f"  • {err}")
    
    if results["warnings"]:
        lines.append(f"\n{Colors.YELLOW}WARNINGS:{Colors.ENDC}")
        for warn in results["warnings"]:
            lines.append(f"  • {warn}")
    
    if results["recommendations"]:
        lines.append(f"\n{Colors.CYAN}RECOMMENDATIONS:{Colors.ENDC}")
        for rec in results["recommendations"]:
            lines.append(f"  • {rec}")
    
    if results["metrics"]:
        lines.append(f"\n{Colors.BLUE}METRICS:{Colors.ENDC}")
        for key, val in results["metrics"].items():
            lines.append(f"  • {key}: {val}")
    
    return "\n".join(lines)


def run_quicktest() -> tuple[bool, dict]:
    """Run the quicktest and return success status and results."""
    print_header("PHASE 1: Running Quicktest")
    
    # Create log directory
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create output directory
    QUICKTEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    log_file = LOG_DIR / f"devstral2_h100_quicktest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    start_time = time.time()
    exit_code, log_content = run_axolotl_train(QUICKTEST_CONFIG, log_file)
    elapsed = time.time() - start_time
    
    print()
    print_header("PHASE 1 Results: Quicktest Analysis")
    
    print_info(f"Duration: {elapsed/60:.1f} minutes")
    print_info(f"Exit code: {exit_code}")
    print()
    
    results = analyze_quicktest_results(log_content, QUICKTEST_OUTPUT_DIR)
    results["exit_code"] = exit_code
    results["duration_seconds"] = elapsed
    
    if exit_code != 0:
        results["success"] = False
        results["errors"].append(f"Training exited with code {exit_code}")
    
    print(generate_recommendations(results))
    
    return results["success"], results


def run_full_training():
    """Run the full training."""
    print_header("PHASE 2: Running Full Training")
    
    # Create output directory
    FULL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    log_file = LOG_DIR / f"devstral2_h100_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    print_info(f"Config: {FULL_CONFIG}")
    print_info(f"Output: {FULL_OUTPUT_DIR}")
    print_info(f"Log: {log_file}")
    print()
    
    start_time = time.time()
    exit_code, log_content = run_axolotl_train(FULL_CONFIG, log_file)
    elapsed = time.time() - start_time
    
    print()
    print_header("PHASE 2 Results: Full Training")
    print_info(f"Duration: {elapsed/3600:.2f} hours")
    print_info(f"Exit code: {exit_code}")
    
    if exit_code == 0:
        print_success("Full training completed successfully!")
    else:
        print_error(f"Full training failed with exit code {exit_code}")
        print_info(f"Check log file: {log_file}")
    
    return exit_code == 0


def main():
    parser = argparse.ArgumentParser(description="Smart training for Devstral-Small-2-24B on H100")
    parser.add_argument("--quicktest-only", "-q", action="store_true", 
                        help="Only run quicktest, don't proceed to full training")
    parser.add_argument("--skip-quicktest", "-s", action="store_true",
                        help="Skip quicktest and go directly to full training")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Proceed to full training even if quicktest fails")
    args = parser.parse_args()
    
    print_header("Smart Training: Devstral-Small-2-24B-Instruct-2512 on H100")
    print(f"{Colors.CYAN}Model:{Colors.ENDC} mistralai/Devstral-Small-2-24B-Instruct-2512")
    print(f"{Colors.CYAN}Format:{Colors.ENDC} FP8 (native on H100)")
    print(f"{Colors.CYAN}Adapter:{Colors.ENDC} LoRA (no additional quantization)")
    print(f"{Colors.CYAN}Dataset:{Colors.ENDC} pankajmathur/OpenThoughts-Agent-v1-SFT")
    print()
    
    # Check GPU compatibility first
    if not check_gpu_compatibility():
        print_error("GPU does not support FP8. Use Devstral-Small-2507 (BF16) with QLoRA instead.")
        return 1
    print()
    
    if not args.skip_quicktest:
        # Run quicktest first
        success, results = run_quicktest()
        
        # Save results
        results_file = LOG_DIR / "devstral2_h100_quicktest_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print_info(f"Results saved to: {results_file}")
        
        if args.quicktest_only:
            print()
            if success:
                print_success("Quicktest passed! Ready for full training.")
                print_info(f"Run: python {__file__} --skip-quicktest")
            else:
                print_error("Quicktest failed. Fix issues before full training.")
            return 0 if success else 1
        
        if not success and not args.force:
            print()
            print_error("Quicktest failed. Fix issues before proceeding.")
            print_info("Use --force to override and proceed anyway.")
            return 1
        
        if not success and args.force:
            print()
            print_warning("Proceeding to full training despite quicktest failures (--force)")
    
    # Run full training
    success = run_full_training()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

