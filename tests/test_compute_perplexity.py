# tests/test_compute_perplexity.py
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from perplexity_calculator import PerplexityCalculator

@pytest.fixture
def mock_perplexity_calculator():
    # Load a small pre-trained model and tokenizer for testing
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return PerplexityCalculator(model, tokenizer)

def test_compute_perplexity(mock_perplexity_calculator):
    text = "This is a test sentence."
    perplexity = mock_perplexity_calculator.compute_perplexity(text)
    
    assert isinstance(perplexity, float), "Perplexity should be a float."
    assert perplexity > 0, "Perplexity should be positive."

def test_compute_perplexities_for_canaries(mock_perplexity_calculator):
    canaries = ["This is a test canary.", "Another test canary."]
    references = ["This is a reference sentence.", "Another reference."]

    canary_perplexities, reference_perplexities = mock_perplexity_calculator.compute_perplexities_for_canaries(canaries, references)
    
    assert isinstance(canary_perplexities, dict), "Canary perplexities should be a dictionary."
    assert isinstance(reference_perplexities, list), "Reference perplexities should be a list."
    assert len(canary_perplexities) == len(canaries), "Each canary should have a perplexity score."
    assert len(reference_perplexities) == len(references), "Each reference should have a perplexity score."
