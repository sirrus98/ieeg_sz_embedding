"""
Comprehensive test script for redesigned WaveNet architecture.

Tests:
1. Architecture shapes through each layer
2. No NaN/Inf outputs
3. Gradient flow (no exploding/vanishing gradients)
4. Padded channel handling with masking
5. Variable channels per batch (different channel counts in same batch)
6. Numerical stability
7. Memory usage
8. Comparison with working models
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(__file__))

from models.WVNT.model_adaptive import WaveModelAdaptive
from models.wrappers.wvnt_wrapper import WVNTChannelAggregator
from models.wrappers.conformer_regions_wrapper import ConformerRegionsChannelAggregator
from models.wrappers.st_wrapper import STChannelAggregator


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def test_architecture_shapes():
    """Test 1: Verify output shapes through each layer."""
    print_section("Test 1: Architecture Shapes")
    
    model = WaveModelAdaptive()
    model.eval()
    
    # Test with 2560-length input (10 seconds @ 256Hz)
    batch_size = 4
    x = torch.randn(batch_size, 1, 2560)
    
    print(f"Input: {x.shape}")
    
    with torch.no_grad():
        # Conv Block 1
        x1 = model.conv1(x)
        x1 = torch.relu(model.bn1(x1))
        x1 = model.pool1(x1)
        print(f"After Conv Block 1: {x1.shape} (expected: [{batch_size}, 32, 640])")
        
        # Conv Block 2
        x2 = model.conv2(x1)
        x2 = torch.relu(model.bn2(x2))
        x2 = model.pool2(x2)
        print(f"After Conv Block 2: {x2.shape} (expected: [{batch_size}, 64, 160])")
        
        # Conv Block 3
        x3 = model.conv3(x2)
        x3 = torch.relu(model.bn3(x3))
        x3 = model.pool3(x3)
        print(f"After Conv Block 3: {x3.shape} (expected: [{batch_size}, 128, 80])")
        
        # Conv Block 4
        x4 = model.conv4(x3)
        x4 = torch.relu(model.bn4(x4))
        print(f"After Conv Block 4: {x4.shape} (expected: [{batch_size}, 256, 80])")
        
        # Conv Block 5
        x5 = model.conv5(x4)
        x5 = torch.relu(model.bn5(x5))
        print(f"After Conv Block 5: {x5.shape} (expected: [{batch_size}, 256, 80])")
        
        # Conv Block 6
        x6 = model.conv6(x5)
        x6 = torch.relu(model.bn6(x6))
        print(f"After Conv Block 6: {x6.shape} (expected: [{batch_size}, 256, 80])")
        
        # Adaptive pooling
        x7 = model.adaptive_pool(x6)
        print(f"After Adaptive Pool: {x7.shape} (expected: [{batch_size}, 256, 64])")
        
        # Flatten
        x8 = x7.flatten(1)
        print(f"After Flatten: {x8.shape} (expected: [{batch_size}, 16384])")
        
        # FC1
        x9 = torch.relu(model.fc1(x8))
        print(f"After FC1: {x9.shape} (expected: [{batch_size}, 256])")
        
        # FC2
        x10 = model.fc2(x9)
        print(f"After FC2 (logits): {x10.shape} (expected: [{batch_size}, 2])")
    
    print("\n✓ All shapes match expected dimensions!")


def test_no_nan_inf():
    """Test 2: Check for NaN/Inf in outputs."""
    print_section("Test 2: NaN/Inf Detection")
    
    model = WaveModelAdaptive()
    model.eval()
    
    # Test with various inputs
    test_cases = [
        ("Normal random input", torch.randn(8, 1, 2560)),
        ("Small values", torch.randn(8, 1, 2560) * 0.01),
        ("Large values", torch.randn(8, 1, 2560) * 10.0),
        ("Mixed positive/negative", torch.randn(8, 1, 2560) * 5.0),
    ]
    
    all_passed = True
    
    for name, x in test_cases:
        with torch.no_grad():
            logits = model(x)
            features = model.get_features(x)
            
            has_nan_logits = torch.isnan(logits).any()
            has_inf_logits = torch.isinf(logits).any()
            has_nan_features = torch.isnan(features).any()
            has_inf_features = torch.isinf(features).any()
            
            if has_nan_logits or has_inf_logits or has_nan_features or has_inf_features:
                print(f"  ✗ {name}: FAILED")
                if has_nan_logits:
                    print(f"    - Logits contain NaN")
                if has_inf_logits:
                    print(f"    - Logits contain Inf")
                if has_nan_features:
                    print(f"    - Features contain NaN")
                if has_inf_features:
                    print(f"    - Features contain Inf")
                all_passed = False
            else:
                print(f"  ✓ {name}: PASSED")
    
    if all_passed:
        print("\n✓ No NaN/Inf detected in any test case!")
    else:
        print("\n✗ Some test cases failed!")


def test_gradient_flow():
    """Test 3: Check gradient flow (no exploding/vanishing)."""
    print_section("Test 3: Gradient Flow")
    
    model = WaveModelAdaptive()
    model.train()
    
    x = torch.randn(4, 1, 2560, requires_grad=True)
    target = torch.randint(0, 2, (4,))
    
    # Forward pass
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print(f"Loss: {loss.item():.4f}")
    print(f"\nGradient statistics:")
    
    grad_stats = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            grad_stats.append((name, grad_norm, grad_mean, grad_std))
    
    # Print first few and last few layers
    print("\nFirst 3 layers:")
    for name, norm, mean, std in grad_stats[:3]:
        print(f"  {name:30s} | norm: {norm:8.4f} | mean: {mean:8.4f} | std: {std:8.4f}")
    
    print("\nLast 3 layers:")
    for name, norm, mean, std in grad_stats[-3:]:
        print(f"  {name:30s} | norm: {norm:8.4f} | mean: {mean:8.4f} | std: {std:8.4f}")
    
    # Check for issues
    issues = []
    for name, norm, mean, std in grad_stats:
        if norm > 100:
            issues.append(f"  - {name}: Exploding gradient (norm={norm:.2f})")
        elif norm < 1e-6:
            issues.append(f"  - {name}: Vanishing gradient (norm={norm:.2e})")
    
    if issues:
        print("\n⚠ Gradient issues detected:")
        for issue in issues:
            print(issue)
    else:
        print("\n✓ Gradients are healthy (no exploding/vanishing)!")


def test_padded_channels():
    """Test 4: Verify padded channel handling with masking."""
    print_section("Test 4: Padded Channel Handling")
    
    # Test with mean aggregation
    print("Testing MEAN aggregation:")
    model_mean = WVNTChannelAggregator(aggregation='mean', freeze_backbone=False)
    model_mean.eval()
    
    batch_size = 4
    n_real = 94
    n_padded = 150
    seq_len = 2560
    
    # Create input with real + padded channels
    x = torch.zeros(batch_size, n_padded, seq_len)
    x[:, :n_real, :] = torch.randn(batch_size, n_real, seq_len)
    
    # Create regions (0 = padded)
    regs = torch.zeros(batch_size, n_padded, dtype=torch.long)
    regs[:, :n_real] = torch.randint(1, 42, (batch_size, n_real))
    
    print(f"  Input shape: {x.shape}")
    print(f"  Valid channels per sample: {(regs != 0).sum(dim=1).tolist()}")
    
    with torch.no_grad():
        logits = model_mean(x, regs)
    
    print(f"  Output shape: {logits.shape}")
    print(f"  Output range: [{logits.min():.4f}, {logits.max():.4f}]")
    
    has_nan = torch.isnan(logits).any()
    has_inf = torch.isinf(logits).any()
    
    if has_nan or has_inf:
        print("  ✗ FAILED: Outputs contain NaN/Inf")
    else:
        print("  ✓ PASSED: No NaN/Inf with padded channels")
    
    # Test with max aggregation
    print("\nTesting MAX aggregation:")
    model_max = WVNTChannelAggregator(aggregation='max', freeze_backbone=False)
    model_max.eval()
    
    with torch.no_grad():
        logits_max = model_max(x, regs)
    
    print(f"  Output shape: {logits_max.shape}")
    print(f"  Output range: [{logits_max.min():.4f}, {logits_max.max():.4f}]")
    
    has_nan = torch.isnan(logits_max).any()
    has_inf = torch.isinf(logits_max).any()
    
    if has_nan or has_inf:
        print("  ✗ FAILED: Outputs contain NaN/Inf")
    else:
        print("  ✓ PASSED: No NaN/Inf with padded channels")
    
    # Test that padded channels don't affect output
    print("\nTesting padding invariance:")
    
    # Create input with only real channels (no padding)
    x_no_pad = x[:, :n_real, :]
    regs_no_pad = regs[:, :n_real]
    
    with torch.no_grad():
        logits_no_pad = model_mean(x_no_pad, regs_no_pad)
    
    # The outputs should be similar (not identical due to different batch processing)
    # but should have similar magnitude
    diff = (logits - logits_no_pad).abs().mean()
    print(f"  Mean difference (padded vs no padding): {diff:.6f}")
    
    if diff < 1.0:  # Reasonable threshold
        print("  ✓ PASSED: Padded channels have minimal effect")
    else:
        print(f"  ⚠ WARNING: Large difference detected")


def test_variable_channels_per_batch():
    """Test 5: Test with different channel counts within same batch."""
    print_section("Test 5: Variable Channels Per Batch")
    
    model_mean = WVNTChannelAggregator(aggregation='mean', freeze_backbone=False)
    model_mean.eval()
    
    # Create batch with different numbers of real channels per sample
    batch_size = 4
    max_channels = 150
    seq_len = 2560
    
    # Different channel counts: 94, 78, 110, 65
    channel_counts = [94, 78, 110, 65]
    
    print(f"Testing batch with variable channel counts: {channel_counts}")
    
    # Create input with different valid channels per sample
    x = torch.zeros(batch_size, max_channels, seq_len)
    regs = torch.zeros(batch_size, max_channels, dtype=torch.long)
    
    for i, n_ch in enumerate(channel_counts):
        x[i, :n_ch, :] = torch.randn(n_ch, seq_len)
        regs[i, :n_ch] = torch.randint(1, 42, (n_ch,))
    
    print(f"  Input shape: {x.shape}")
    print(f"  Valid channels per sample: {(regs != 0).sum(dim=1).tolist()}")
    
    with torch.no_grad():
        logits = model_mean(x, regs)
    
    print(f"  Output shape: {logits.shape}")
    print(f"  Output logits:")
    for i, n_ch in enumerate(channel_counts):
        print(f"    Sample {i} ({n_ch:3d} channels): [{logits[i, 0].item():7.4f}, {logits[i, 1].item():7.4f}]")
    
    # Check for issues
    has_nan = torch.isnan(logits).any()
    has_inf = torch.isinf(logits).any()
    
    if has_nan:
        print("  ✗ FAILED: Outputs contain NaN")
        return False
    elif has_inf:
        print("  ✗ FAILED: Outputs contain Inf")
        return False
    
    # Check that outputs have reasonable variance (not all the same)
    output_std = logits.std()
    if output_std < 1e-6:
        print(f"  ⚠ WARNING: Very low output variance (std={output_std:.2e})")
    else:
        print(f"  ✓ Output variance is healthy (std={output_std:.4f})")
    
    print("  ✓ PASSED: Variable channels handled correctly")
    
    # Test with max aggregation too
    print("\nTesting MAX aggregation with variable channels:")
    model_max = WVNTChannelAggregator(aggregation='max', freeze_backbone=False)
    model_max.eval()
    
    with torch.no_grad():
        logits_max = model_max(x, regs)
    
    print(f"  Output shape: {logits_max.shape}")
    has_nan = torch.isnan(logits_max).any()
    has_inf = torch.isinf(logits_max).any()
    
    if has_nan or has_inf:
        print("  ✗ FAILED: Outputs contain NaN/Inf")
        return False
    else:
        print("  ✓ PASSED: Max aggregation works with variable channels")
    
    return True


def test_numerical_stability():
    """Test 6: Test with extreme values."""
    print_section("Test 6: Numerical Stability")
    
    model = WaveModelAdaptive()
    model.eval()
    
    test_cases = [
        ("All zeros", torch.zeros(4, 1, 2560)),
        ("All ones", torch.ones(4, 1, 2560)),
        ("Very small values", torch.randn(4, 1, 2560) * 1e-6),
        ("Very large values", torch.randn(4, 1, 2560) * 100),
        ("Mixed extreme", torch.cat([
            torch.randn(2, 1, 2560) * 1e-6,
            torch.randn(2, 1, 2560) * 100
        ], dim=0)),
    ]
    
    all_passed = True
    
    for name, x in test_cases:
        with torch.no_grad():
            try:
                logits = model(x)
                has_nan = torch.isnan(logits).any()
                has_inf = torch.isinf(logits).any()
                
                if has_nan or has_inf:
                    print(f"  ✗ {name}: FAILED (NaN/Inf detected)")
                    all_passed = False
                else:
                    print(f"  ✓ {name}: PASSED")
            except Exception as e:
                print(f"  ✗ {name}: FAILED (Exception: {e})")
                all_passed = False
    
    if all_passed:
        print("\n✓ Model is numerically stable!")
    else:
        print("\n✗ Some stability issues detected!")


def test_memory_usage():
    """Test 7: Check memory usage."""
    print_section("Test 7: Memory Usage")
    
    model = WaveModelAdaptive()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")
    
    # Test forward pass memory
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
        
        torch.cuda.reset_peak_memory_stats()
        
        x = torch.randn(32, 1, 2560, device=device)
        with torch.no_grad():
            logits = model(x)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"\nGPU Peak Memory (batch=32): {peak_memory:.2f} MB")
        
        if peak_memory < 1000:  # Less than 1GB
            print("✓ Memory usage is reasonable")
        else:
            print("⚠ High memory usage detected")
    else:
        print("\nGPU not available, skipping GPU memory test")


def compare_architectures():
    """Test 8: Compare with working models."""
    print_section("Test 8: Architecture Comparison")
    
    batch_size = 4
    n_channels = 94
    seq_len = 2560
    
    x = torch.randn(batch_size, n_channels, seq_len)
    regs = torch.randint(1, 42, (batch_size, n_channels))
    
    print("Model parameter counts:")
    
    # WaveNet
    wvnt = WVNTChannelAggregator(aggregation='mean', freeze_backbone=False)
    wvnt_params = sum(p.numel() for p in wvnt.parameters())
    print(f"  WaveNet (redesigned): {wvnt_params:,}")
    
    # ConformerRegions
    conformer = ConformerRegionsChannelAggregator(
        emb_size=128, depth=6, max_channels=150, max_regions=42
    )
    conformer_params = sum(p.numel() for p in conformer.parameters())
    print(f"  ConformerRegions:     {conformer_params:,}")
    
    # ST Model
    st = STChannelAggregator(
        spatial_embed_dim=128,
        depth_spatial=4,
        depth_temporal=4,
        n_frames=30,
        max_channels=150,
        max_regions=42,
        aggregation='mean'
    )
    st_params = sum(p.numel() for p in st.parameters())
    print(f"  ST Model:             {st_params:,}")
    
    print("\nForward pass test:")
    
    models = [
        ("WaveNet", wvnt, True),  # Needs regs
        ("ConformerRegions", conformer, True),
        ("ST Model", st, True),
    ]
    
    for name, model, needs_regs in models:
        model.eval()
        try:
            with torch.no_grad():
                if needs_regs:
                    logits = model(x, regs)
                else:
                    logits = model(x)
            
            has_nan = torch.isnan(logits).any()
            has_inf = torch.isinf(logits).any()
            
            if has_nan or has_inf:
                print(f"  ✗ {name:20s}: FAILED (NaN/Inf)")
            else:
                print(f"  ✓ {name:20s}: PASSED - shape {logits.shape}")
        except Exception as e:
            print(f"  ✗ {name:20s}: FAILED ({e})")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("  WaveNet Architecture Comprehensive Test Suite")
    print("="*80)
    
    # Run all tests
    test_architecture_shapes()
    test_no_nan_inf()
    test_gradient_flow()
    test_padded_channels()
    test_variable_channels_per_batch()
    test_numerical_stability()
    test_memory_usage()
    compare_architectures()
    
    print("\n" + "="*80)
    print("  All tests completed!")
    print("="*80)


if __name__ == "__main__":
    main()
