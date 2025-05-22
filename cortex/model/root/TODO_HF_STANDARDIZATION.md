# TODO: HuggingFace Parameter Name Standardization

## Issue
Multiple components use custom parameter names instead of standard HuggingFace names.

## Root Nodes (TransformerRootV2/V3)
- `tgt_tok_idxs` → `input_ids`
- `padding_mask` → `attention_mask`
- Other root parameters as needed

## Leaf Nodes (ClassifierLeaf, etc.)
- `targets` → `labels` (standard HF convention for classification tasks)
- Verify other leaf node parameter naming

## Goal
Standardize to HuggingFace naming conventions across all components

## Benefits
- Better compatibility with HuggingFace ecosystem
- More intuitive for users familiar with transformers
- Cleaner integration with HuggingFace tokenizers and models

## Implementation Plan
1. Update TransformerRootV2/V3 forward method signatures
2. Add backward compatibility aliases
3. Update all tests and examples
4. Update documentation

## Priority
Medium - implement after core Lightning integration is complete
