#!/usr/bin/env python3
import os
import sys

# Set cache directories
os.environ['HF_HOME'] = '/fsxl/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/fsxl/.cache/huggingface'
os.environ['HF_DATASETS_OFFLINE'] = '1'

print(f"HF_HOME: {os.environ.get('HF_HOME')}")
print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE')}")

# Check if cache directory exists
cache_dir = '/fsxl/.cache/huggingface'
if os.path.exists(cache_dir):
    print(f"✓ Cache directory exists: {cache_dir}")
    print(f"  Contents: {os.listdir(cache_dir)}")
else:
    print(f"✗ Cache directory NOT found: {cache_dir}")
    sys.exit(1)

# Try to load tokenizer
try:
    from transformers import AutoTokenizer
    print("Attempting to load tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        'hf-internal-testing/llama-tokenizer',
        legacy=False,
        local_files_only=True
    )
    print(f"✓ SUCCESS: Tokenizer loaded!")
    print(f"  Vocab size: {tokenizer.vocab_size}")
except Exception as e:
    print(f"✗ FAILED to load tokenizer: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)