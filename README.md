# ProofPilot — AI Dispute Risk Manager

AI Risk Manager for Razorpay merchant chargeback evidence readiness, built for the Razorpay AI Buildathon (Track 02: AI Risk Manager).

## Overview

ProofPilot evaluates the strength of merchant evidence before a chargeback response is submitted to payment networks. It detects evidence quality (present, weak, missing, irrelevant), produces a confidence-gated evidence-readiness score, calculates counterfactual risk improvements, and routes disputes to auto-drafting, evidence collection, or human analyst review.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Regenerate Dataset

```bash
python data_generator.py --cases 60 --test-ratio 0.33 --out outputs/synthetic_chargeback_dataset.json
```

## Run Scorer

```bash
python scoring/scorer.py
```

## Evaluate Benchmark

```bash
python eval/evaluate.py
```

## Features

- **Razorpay Dispute Workflow Simulation**: Uses Razorpay-aligned payment IDs, order IDs, dispute IDs (`disp_...`), amounts in INR paise, and webhook payload structures.
- **Evidence Extraction**: Parses unstructured document text and detects status (`present`, `weak`, `missing`).
- **Counterfactual Risk Improvement**: Shows projected readiness score gains and risk tier upgrades if specific missing evidence is uploaded.
- **Confidence-Gated AI Routing**: Prevents false-positive auto-drafts by requiring $\ge 80\%$ completeness and $\ge 75\%$ confidence.
- **LLM Reasoning & Gap Explanation**: Generates precise advice on what evidence to retrieve, with an optional Gemini / OpenAI / Groq LLM integration (and local fallback).
