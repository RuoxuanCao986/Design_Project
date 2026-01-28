#!/usr/bin/env python3
"""
Test script to debug the generate_text method
Supports command line arguments to specify models
"""

import sys
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llama_demos.model_config_manager import ModelConfigManager

def generate_text_local(model_name, prompt, max_tokens=30):
    """
    Generate text using local model loading
    """
    try:
        print(f"\n🔄 Generating text with {model_name}...")
        print(f"   Prompt: {prompt}")
        
        # Debug CUDA initialization
        print("   🛠️ CUDA Debug Info:")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        print(f"   CUDA device count: {torch.cuda.device_count()}")
        try:
            print(f"   CUDA version: {torch.version.cuda}")
        except Exception as e:
            print(f"   CUDA version error: {e}")
        
        # Try to force CUDA initialization
        try:
            torch.cuda.init()
            print("   CUDA initialized successfully")
            print(f"   CUDA device count after init: {torch.cuda.device_count()}")
        except Exception as e:
            print(f"   CUDA init error: {e}")
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Using device: {device}")
        
        # Load tokenizer
        print(f"   1. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"   ✅ Tokenizer loaded successfully")
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"   Set pad_token to eos_token: {tokenizer.pad_token}")
        
        # Load model
        print(f"   2. Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        print(f"   ✅ Model loaded successfully")
        
        # Move to device
        if device == "cpu":
            model = model.to(device)
        
        model.eval()
        
        # Encode prompt
        print(f"   3. Encoding prompt...")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        print(f"   ✅ Prompt encoded successfully")
        
        # Generate text with minimal parameters
        print(f"   4. Generating text...")
        print(f"   Using minimal parameters: max_new_tokens={max_tokens}, no sampling")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        print(f"   ✅ Text generation completed")
        print(f"   Output shape: {outputs.shape}")
        
        # Decode generated text
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print(f"   ✅ Text decoded successfully")
        
        print(f"\nResult: {generated_text}")
        print("✅ Success!")
        return generated_text
        
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_generate_text():
    """
    Test direct model loading without model manager
    """
    parser = argparse.ArgumentParser(description="Test text generation with different models")
    parser.add_argument("model", nargs="*", help="Model nickname(s) to test (e.g., llama-3.2-1b deepseek-chat)")
    parser.add_argument("--max-tokens", type=int, default=30, help="Maximum tokens to generate")
    parser.add_argument("--prompt", type=str, default="Please explain what artificial intelligence is", help="Test prompt")
    args = parser.parse_args()
    
    # Get model list from command line or use default
    if args.model:
        model_nicknames = args.model
    else:
        # Default to a small model for quick testing
        model_nicknames = ["facebook/opt-1.3b"]
    
    # Initialize model config manager
    config_manager = ModelConfigManager()
    
    print("Testing direct model loading...")
    print("=" * 80)
    print(f"Test prompt: {args.prompt}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Models to test: {model_nicknames}")
    print("=" * 80)
    
    # Convert nicknames to model identifiers
    test_models = []
    for nickname in model_nicknames:
        model_info = config_manager.get_model_info_by_nickname(nickname)
        if model_info:
            model_identifier = model_info["model_identifier"]
            test_models.append((nickname, model_identifier))
            print(f"✓ {nickname} -> {model_identifier}")
        else:
            # Try to use the nickname directly as a model identifier
            test_models.append((nickname, nickname))
            print(f"? {nickname} (using as direct identifier)")
    
    print("=" * 80)
    
    # Test each model
    for nickname, model_identifier in test_models:
        print(f"\n{'='*80}")
        print(f"Testing: {nickname}")
        print(f"Model: {model_identifier}")
        print(f"{'='*80}")
        try:
            text = generate_text_local(model_identifier, args.prompt, args.max_tokens)
            if text:
                print("✅ Success!")
            else:
                print("❌ Failed to generate text")
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    print("\n" + "=" * 80)
    print("Test completed!")

if __name__ == "__main__":
    test_generate_text()
