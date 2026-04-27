# Emotion MI

Replication of Anthropic's mechanistic interpretability research on emotion concepts in LLMs.

**Paper**: [Emotion concepts emerge in large language models as functional analogues to human feelings](https://www.anthropic.com/research/emotion-concepts-function)

---

## Pipeline Overview

```
1. Generate Stimuli       → Claude API generates narratives for 171 emotion concepts
2. Extract Activations    → TransformerLens hooks into residual stream (open-source model)
3. Vector Analysis        → PCA map, cosine similarity, nearest-neighbor structure
4. Linear Probing         → Validate vectors generalize to held-out corpora
5. Steering Experiments   → Causal intervention: amplify/suppress emotion vectors
6. Behavioral Testing     → Correlation between emotion activation and task preference
```

## Setup

```bash
# Install dependencies (requires Python 3.11+)
pip install -e .

# Or with uv:
uv sync

# Copy and fill in credentials
cp .env.example .env
```

## Key Design Decision

The paper used **Claude Sonnet 4.5** with proprietary activation access. This replication uses:
- **Claude API** for narrative generation (Step 1) and behavioral testing (Step 6)
- **Open-source models via TransformerLens** for activation extraction and steering (Steps 2–5)

Start with `gpt2-xl` for fast iteration, then upgrade to `meta-llama/Llama-3.1-8B` for quality results.

## Run Order

```bash
# Notebooks (recommended for exploration)
jupyter lab notebooks/

# Or scripts end-to-end
python src/generate_stimuli.py   # writes data/stimuli/narratives.jsonl
python src/extract_activations.py  # writes results/vectors/
python src/emotion_vectors.py    # writes results/vectors/pca_map.png
```

## Directory Structure

```
data/
  emotions.txt          # 171 emotion concept labels
  stimuli/              # Generated Claude narratives (jsonl)
  corpora/              # Validation text corpora
src/
  generate_stimuli.py   # Step 1
  extract_activations.py  # Step 2
  emotion_vectors.py    # Step 3
  probing.py            # Step 4
  steering.py           # Step 5
  behavioral.py         # Step 6
notebooks/
  01_generate_and_extract.ipynb
  02_vector_analysis.ipynb
  03_probing.ipynb
  04_steering.ipynb
results/
  vectors/              # emotion_matrix.npy, emotion_labels.json
  probing/
  steering/
  behavioral/
```
