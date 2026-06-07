# InvestIQ — Behavioral Finance Engine

> **Detecting cognitive biases in retail investor behavior using machine learning, and nudging them toward better decisions.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

## Problem Statement

**40% of non-investors** don't invest because they simply don't know how (World Economic Forum). Existing platforms focus on *what to buy* — nobody helps investors understand *why they make bad decisions*.

InvestIQ is a **behavioral finance research project** that:
1. **Detects cognitive biases** (FOMO, loss aversion, overconfidence) from trading patterns using ML classifiers
2. **Analyzes market sentiment** using FinBERT NLP on financial news
3. **Generates personalized nudges** to counter detected biases in real-time

## Research Foundation

Built on established behavioral finance theory:
- **Prospect Theory** (Kahneman & Tversky, 1979) — Loss aversion framework
- **Nudge Theory** (Thaler & Sunstein, 2008) — Choice architecture for better decisions
- **Behavioral Portfolio Theory** (Shefrin & Statman, 2000) — Mental accounting in portfolios

## Project Structure

```
InvestIQ/
├── data/                    # Datasets (synthetic + real market data)
├── models/                  # Trained model artifacts
├── notebooks/               # Research notebooks with analysis
├── src/
│   ├── bias_detection/      # ML pipeline for cognitive bias classification
│   ├── sentiment/           # FinBERT-based market sentiment engine
│   └── nudge_engine/        # Personalized nudge generation system
├── results/                 # Evaluation metrics, plots, reports
├── tests/                   # Unit tests
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/En1gma1708/InvestIQ.git
cd InvestIQ

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Generate synthetic dataset
python src/bias_detection/data_generator.py

# Train bias detection model
python src/bias_detection/train.py

# Evaluate results
python src/bias_detection/evaluate.py
```

## Phase 1: Bias Detection Model

Classifies investor behavior into cognitive bias categories:

| Bias Type | Description | Key Features |
|-----------|-------------|-------------|
| **FOMO / Herd** | Buying because everyone else is | Trade correlation with market trends, social media volume |
| **Loss Aversion** | Holding losers too long, selling winners too early | Hold duration asymmetry, stop-loss behavior |
| **Overconfidence** | Excessive trading, concentrated positions | Trade frequency, sector concentration, position sizing |
| **Recency Bias** | Over-weighting recent events | Portfolio churn rate, reaction to recent news |
| **Anchoring** | Fixating on purchase price | Sell decisions relative to buy price |

## Results

*Results will be updated as experiments complete.*

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

**Sahil Sharma** — [@En1gma1708](https://github.com/En1gma1708)
