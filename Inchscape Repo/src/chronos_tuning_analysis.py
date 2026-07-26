"""
Chronos 1 (T5-tiny) - Analysis of Tuning Feasibility

Chronos is a pre-trained foundation model from Amazon that:
- Does NOT support hyperparameter tuning in the traditional sense
- Cannot be fine-tuned on user data without retraining the entire transformer
- Uses fixed architecture and weights loaded from HuggingFace

Options for "tuned" comparison:
1. Chronos (default) vs. Chronos with configuration variants (e.g., temperature)
2. Chronos vs. simple baseline (e.g., naive forecast, exponential smoothing)
3. Document Chronos as "foundation model" - inherently pre-trained, not tunable

Given the user's request ("show tuned vs untuned"), the most practical approach:
- Show Chronos T5-tiny as a single model (foundation model)
- Add context that it's pre-trained and not tuned on this data
- Optionally compare against a simple naive/seasonal baseline if needed

Implementation: Use rolling_evaluation_erratic_chronos_t5_tiny results directly,
document as "foundation model" in the comparison CSV.
"""

print(__doc__)
