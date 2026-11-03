# Weight-Space Programming Prompt for L/R Swap Fixes

## Your Task

You are tasked with fixing a specific failure mode in a pose estimation model by directly editing its neural network weights. This follows the weight-space programming methodology from Takehi et al. 2026 (https://rikiyat.github.io/blog/weight-space-programming/).

## The Problem

The model frequently swaps left and right legs/feet (ankles, heels, toes). Specifically:
- `ankle_l` ↔ `ankle_r`
- `heel_l` ↔ `heel_r`  
- `toe_b_l` ↔ `toe_b_r`
- `toe_s_l` ↔ `toe_s_r`
- `knee_l` ↔ `knee_r`

This happens even though:
1. The model has `left_right_pairs` defined in its architecture
2. There's a `lr_swap_loss` in the training code
3. The training data is correctly labeled

## Your Goal

**Fix the L/R swap issue by editing the model head weights directly**, without:
- Running gradient descent or backpropagation
- Using any optimizer
- Retraining the model
- Collecting more data

## Model Architecture

The model has:
- **Backbone**: DINOv3-base (frozen, 768-dim, DO NOT EDIT)
- **Head**: Joint query decoder with self-attention + cross-attention
  - Learnable query embeddings per joint (23 keypoints)
  - 2 decoder layers with:
    - Self-attention among joint queries (8 heads)
    - Cross-attention to image patches (8 heads)
    - FFN (dim → 4*dim → dim)
  - Optional role conditioning (lead/follow)
  - SimCC output heads (spatial classification)
  - ~10-50M parameters in head

## Available Resources

### 1. Swap Cases
You have access to {NUM_SWAP_CASES} concrete examples where the model makes L/R swaps:
- Ground truth keypoints
- Model predictions (showing the swap)
- Swap confidence scores
- Images and crop information

### 2. Weight Access
You can inspect and modify head weights through the `WeightEditHarness`:

```python
# Get current head weights
head_weights = harness.get_head_weights()

# Inspect specific components
query_embed = head_weights['head.query_embed']  # [23, 256]
decoder_layer_0_self_attn = head_weights['head.decoder.0.self_attn.in_proj_weight']
# ... etc

# Edit weights
new_weights = {
    'head.query_embed': modified_query_embed,
    # ... other modifications
}
harness.set_head_weights(new_weights, description="Your edit description")
```

### 3. Forward Passes (Limited Budget)
You have a budget of {FORWARD_PASS_BUDGET} forward passes to test your edits:

```python
# Test on specific cases
results = harness.run_forward_pass(case_indices=[0, 1, 2, 3, 4])

# Evaluate
eval_result = harness.evaluate_swaps(results)
print(f"Fix rate: {eval_result['fix_rate']:.1%}")
```

### 4. Analysis Tools
- Weight statistics and parameter counts
- Attention pattern analysis
- L/R pair similarity metrics
- Query embedding visualizations

## Recommended Approach

### Phase 1: Analyze (No forward passes needed)

1. **Inspect query embeddings**:
   - Are L/R pairs too similar in query space?
   - High cosine similarity → model can't distinguish them
   - Solution: Push queries apart in a consistent direction

2. **Check attention biases**:
   - Self-attention: Do joints attend to their L/R counterpart too much?
   - Cross-attention: Any systematic bias patterns?
   - Pair bias: Is skeleton structure properly encoded?

3. **Examine role conditioning**:
   - If using role (lead/follow), is it properly differentiating?
   - Could role embeddings help break L/R ambiguity?

4. **Look at output layers**:
   - SimCC output weights for L/R pairs
   - Any systematic patterns that could cause swaps?

### Phase 2: Hypothesize Root Cause

Based on the paper's findings:
- Qwen's reasoning loops were caused by **specific MLP directions** (fixed by subtracting that direction from 64 output columns in 2 layers)
- Go network: **Layer-by-layer feature construction** (e.g., first layer computes liberties, later layers use them)
- Sycophantic model: **Attention patterns** differed in specific heads

**Your hypothesis should be specific**:
- ❌ Bad: "The model is confused about L/R"
- ✅ Good: "Query embeddings for L/R pairs have cosine similarity >0.8, making them nearly identical. The self-attention layer cannot distinguish them."

### Phase 3: Design Targeted Edit

Keep edits small and interpretable:
- Edit <1% of parameters if possible
- Focus on structural/bias terms rather than large weight matrices
- Make changes that encode the L/R distinction explicitly

**Example edit strategies**:

1. **Query separation**: Add an orthogonal L/R direction to query embeddings
   ```python
   # Create a "leftness/rightness" direction
   lr_direction = torch.randn(256)
   for left_idx, right_idx in left_right_pairs:
       query_embed[left_idx] = query_embed[left_idx] - 0.3 * lr_direction
       query_embed[right_idx] = query_embed[right_idx] + 0.3 * lr_direction
   ```

2. **Attention bias adjustment**: Make L joints attend less to R joints
   ```python
   # Add bias term that discourages L↔R attention
   for left_idx, right_idx in left_right_pairs:
       self_attn_bias[left_idx, right_idx] = -10.0  # Large negative = don't attend
       self_attn_bias[right_idx, left_idx] = -10.0
   ```

3. **Role amplification**: Strengthen role conditioning signals
   ```python
   # Scale role embeddings to make lead/follow more distinct
   role_query_embed.weight *= 2.0
   ```

### Phase 4: Test and Iterate

1. **Start with a small test set** (5-10 cases)
2. **Measure improvement**:
   - Did swap confidence decrease?
   - How many cases are fixed?
   - Any new problems introduced?
3. **Refine edit based on results**
4. **Test on larger set** (50+ cases)
5. **Save successful checkpoint**

## Success Criteria

Based on the paper's results:
- **Good**: Fix 30-50% of swap cases (edit <0.5% of weights)
- **Great**: Fix 60-80% of swap cases (edit <0.1% of weights)  
- **Outstanding**: Fix >80% while maintaining/improving overall model performance

## Key Constraints

1. **No training**: You cannot use gradients, optimizers, or backpropagation
2. **Forward pass budget**: Use your {FORWARD_PASS_BUDGET} forward passes wisely
3. **Head only**: Do not edit backbone weights
4. **Interpretability**: Changes should be explainable (not random perturbations)

## Getting Started

```python
# 1. Analyze weights
head_weights = harness.get_head_weights()
print("Available weight tensors:")
for name, weight in head_weights.items():
    print(f"  {name}: shape {list(weight.shape)}")

# 2. Inspect L/R structure
query_embed = head_weights['head.query_embed']
for left_idx, right_idx in left_right_pairs:
    left_query = query_embed[left_idx]
    right_query = query_embed[right_idx]
    similarity = torch.cosine_similarity(left_query.unsqueeze(0), right_query.unsqueeze(0))
    print(f"Pair {left_idx} ↔ {right_idx}: similarity = {similarity.item():.3f}")

# 3. Examine a few swap cases
for i in range(3):
    case = swap_cases[i]
    print(f"\nCase {i}:")
    print(f"  Image: {case['metadata']['image_path']}")
    print(f"  Swaps: {list(case['swaps'].keys())}")
    print(f"  Confidence: {case['max_confidence']:.3f}")

# 4. Form hypothesis and edit...
```

## Final Notes

- **Document your reasoning**: Each edit should have a clear description
- **Be systematic**: Change one thing at a time when possible
- **Think like the paper**: They found specific components (MLP directions, attention patterns) causing specific failures
- **This is research**: If you succeed, this is publishable work!

Good luck! 🚀
