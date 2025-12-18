#!/usr/bin/env python3
"""
Direct LoRA merge script without axolotl CLI dependencies.
Merges a LoRA adapter with its base model and saves the result.
"""

import argparse
import os
import torch
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoConfig
from peft import PeftModel
from huggingface_hub import HfApi, upload_folder

# Use mistral tokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


def load_mistral_tokenizer(model_path: str, cache_dir: str = None):
    """Load Mistral tokenizer from model path."""
    from huggingface_hub import hf_hub_download
    
    # Download tekken.json
    tokenizer_path = hf_hub_download(
        repo_id=model_path,
        filename="tekken.json",
        cache_dir=cache_dir,
    )
    
    tokenizer = MistralTokenizer.from_file(tokenizer_path)
    return tokenizer, tokenizer_path


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--base-model", type=str, default="mistralai/Devstral-Small-2507",
                        help="Base model HuggingFace ID")
    parser.add_argument("--adapter-path", type=str, 
                        default="/home/ubuntu/us-east-1-nano-chat-exp/outputs/Devstral-Small-2507-sft-v1/qlora-out",
                        help="Path to LoRA adapter")
    parser.add_argument("--output-dir", type=str,
                        default="/home/ubuntu/us-east-1-nano-chat-exp/outputs/Devstral-Small-2507-sft-v1/merged",
                        help="Output directory for merged model")
    parser.add_argument("--upload-repo", type=str, default=None,
                        help="HuggingFace repo ID to upload to (e.g., 'username/model-name')")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Data type for the merged model")
    parser.add_argument("--device-map", type=str, default="auto",
                        help="Device map for loading model")
    args = parser.parse_args()
    
    # Set dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]
    
    print(f"=== LoRA Merge Script ===")
    print(f"Base model: {args.base_model}")
    print(f"Adapter path: {args.adapter_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Data type: {args.dtype}")
    print()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load base model config
    print("Loading model configuration...")
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    
    # Load base model
    print(f"Loading base model: {args.base_model}")
    print("This may take a while for a 24B model...")
    
    # Check if we should use CPU (for large models on small GPUs)
    use_cpu = args.device_map == "cpu"
    
    if use_cpu:
        print("Using CPU for model loading (this will be slower but works for large models)")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch_dtype,
            device_map=None,  # Load on CPU
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch_dtype,
            device_map=args.device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    
    if torch.cuda.is_available() and not use_cpu:
        print(f"Base model loaded. GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    else:
        print("Base model loaded on CPU.")
    
    # Load LoRA adapter
    print(f"Loading LoRA adapter from: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    
    print("LoRA adapter loaded. Starting merge...")
    
    # Merge and unload
    model = model.merge_and_unload(progressbar=True)
    
    print("Merge complete!")
    
    # Set model to use cache for inference
    model.config.use_cache = True
    if hasattr(model, 'generation_config'):
        model.generation_config.do_sample = True
    
    # Save merged model
    print(f"Saving merged model to: {args.output_dir}")
    model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size="10GB",
    )
    
    # Copy tokenizer files from base model or adapter
    print("Copying tokenizer files...")
    from huggingface_hub import hf_hub_download
    import shutil
    
    # Download and copy tokenizer files from base model
    tokenizer_files = ["tekken.json", "config.json", "generation_config.json"]
    for filename in tokenizer_files:
        try:
            src_file = hf_hub_download(repo_id=args.base_model, filename=filename)
            dst_file = output_path / filename
            if not dst_file.exists():
                shutil.copy(src_file, dst_file)
                print(f"  Copied {filename}")
        except Exception as e:
            print(f"  Skipping {filename}: {e}")
    
    print(f"\nMerged model saved to: {args.output_dir}")
    
    # Upload if requested
    if args.upload_repo:
        print(f"\nUploading to HuggingFace: {args.upload_repo}")
        api = HfApi()
        
        # Create repo if it doesn't exist
        try:
            api.create_repo(repo_id=args.upload_repo, exist_ok=True, private=False)
        except Exception as e:
            print(f"Note: {e}")
        
        # Upload folder
        api.upload_folder(
            folder_path=args.output_dir,
            repo_id=args.upload_repo,
            commit_message="Upload merged LoRA model",
        )
        print(f"Upload complete: https://huggingface.co/{args.upload_repo}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
