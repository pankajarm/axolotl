#!/usr/bin/env python3
"""
Preprocess OpenThoughts-Agent-v1-SFT dataset to filter problematic examples.

Problem: Some assistant messages have empty content and no tool_calls,
which violates the Mistral tokenizer's validation rules.

This script filters out those problematic examples and saves a clean dataset locally.
"""

from datasets import load_dataset
import os

def has_valid_assistant_messages(example):
    """
    Check if all assistant messages in the conversation are valid.
    Valid means: has non-empty content OR has tool_calls (but not neither).
    """
    conversations = example.get("conversations", [])
    
    for msg in conversations:
        if msg.get("role") == "assistant":
            has_content = bool(msg.get("content"))
            has_tool_calls = bool(msg.get("tool_calls"))
            
            # Invalid: neither content nor tool_calls
            if not has_content and not has_tool_calls:
                return False
            
            # Also invalid for Mistral: both content AND tool_calls
            # (uncomment if needed, but our dataset doesn't have this issue)
            # if has_content and has_tool_calls:
            #     return False
    
    return True


def preprocess_dataset():
    print("Loading dataset from HuggingFace...")
    ds = load_dataset("open-thoughts/OpenThoughts-Agent-v1-SFT", split="train")
    
    original_size = len(ds)
    print(f"Original dataset size: {original_size}")
    
    # Filter out problematic examples
    print("Filtering problematic examples...")
    ds_filtered = ds.filter(
        has_valid_assistant_messages,
        num_proc=4,
        desc="Filtering invalid assistant messages"
    )
    
    filtered_size = len(ds_filtered)
    removed_count = original_size - filtered_size
    
    print(f"Filtered dataset size: {filtered_size}")
    print(f"Removed {removed_count} problematic examples ({removed_count/original_size*100:.4f}%)")
    
    # Save locally
    output_dir = "/lambda/nfs/us-east-1-nano-chat-exp/dataset/openthoughts-agent-sft-clean"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as parquet for efficient loading
    output_path = os.path.join(output_dir, "train.parquet")
    print(f"Saving filtered dataset to {output_path}...")
    ds_filtered.to_parquet(output_path)
    
    # Also save as JSON for inspection
    json_path = os.path.join(output_dir, "train.json")
    print(f"Saving as JSON to {json_path}...")
    ds_filtered.to_json(json_path)
    
    print("\n✅ Done! Dataset is now ready for training.")
    print(f"\nUpdate your YAML config to use the local dataset:")
    print(f"  path: {output_dir}")
    print(f"  or")
    print(f"  data_files: {output_path}")
    
    return ds_filtered


if __name__ == "__main__":
    preprocess_dataset()
