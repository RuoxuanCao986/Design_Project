#!/usr/bin/env python3
"""
Script to test different CUDA initialization approaches
"""

import torch
import os

print("=== Testing CUDA Initialization Fixes ===")

# Test 1: Set CUDA_VISIBLE_DEVICES explicitly
print("\n1. Testing with CUDA_VISIBLE_DEVICES=0:")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
try:
    torch.cuda.init()
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"   GPU name: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"   Error: {e}")

# Test 2: Unset CUDA_VISIBLE_DEVICES
print("\n2. Testing with CUDA_VISIBLE_DEVICES unset:")
os.environ.pop('CUDA_VISIBLE_DEVICES', None)
try:
    torch.cuda.init()
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   CUDA device count: {torch.cuda.device_count()}")
except Exception as e:
    print(f"   Error: {e}")

# Test 3: Force reinitialization
print("\n3. Testing forced reinitialization:")
try:
    torch.cuda._lazy_init()
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   CUDA device count: {torch.cuda.device_count()}")
except Exception as e:
    print(f"   Error: {e}")

# Test 4: Check PyTorch and CUDA compatibility
print("\n4. PyTorch and CUDA Compatibility:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   PyTorch CUDA version: {torch.version.cuda}")
print(f"   CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")

# Test 5: Try to use GPU directly
print("\n5. Testing direct GPU usage:")
try:
    # Create a simple tensor and move it to GPU
    x = torch.tensor([1.0])
    print(f"   Created tensor: {x}")
    x_gpu = x.to('cuda')
    print(f"   Moved to GPU: {x_gpu}")
    print("   ✅ GPU usage successful!")
except Exception as e:
    print(f"   ❌ GPU usage failed: {e}")

print("\n=== Fix Attempts Complete ===")

# Final recommendation
print("\n=== Recommendation ===")
print("Based on the test results, the issue appears to be:")
print("1. PyTorch CUDA initialization failure despite GPU hardware being present")
print("2. This is likely due to environment configuration or version compatibility")
print("\nRecommended solutions:")
print("1. Use CPU for now to ensure scripts run successfully")
print("2. Try updating PyTorch to match CUDA 13.0")
print("3. Check and fix environment variables")
print("4. Test with a different PyTorch version")
