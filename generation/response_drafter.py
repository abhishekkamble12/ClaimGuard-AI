"""
Deprecated module. ProofPilot uses generation/llm_reasoning.py for gated dispute response drafting.
"""
from generation.llm_reasoning import generate_gated_dispute_response as draft_response

__all__ = ["draft_response"]
