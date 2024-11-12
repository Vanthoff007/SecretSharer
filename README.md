# The Secret Sharer Experiment
## 1. Motivation
Large Language Models (LLMs) trained on massive datasets have shown remarkable capabilities across various tasks. However, as demonstrated by Carlini et al. in their work *["The Secret Sharer"](https://arxiv.org/abs/2012.07805)*, these models are prone to unintended memorization, where rare or sensitive sequences from the training data are memorized and can potentially be leaked. This poses critical privacy risks, particularly when models are deployed in real-world applications, such as predictive text systems or chatbots, which may inadvertently expose sensitive information like social security numbers, credit card details, or private correspondence.

The ability to detect and quantify this unintended memorization is crucial for:
- Enhancing privacy-preserving machine learning models.
- Ensuring compliance with privacy laws such as [GDPR](https://gdpr.eu/).
- Encouraging the development of more generalizable models that avoid overfitting on rare data.

## 2. Introduction to Unintended Memorization
Machine learning models, especially generative models, can sometimes unintentionally memorize parts of their training data. This project builds on *The Secret Sharer* paper, which describes a testing methodology to assess how models might memorize rare sequences, or “secrets,” in the training data. Such memorization becomes critical when sensitive data is involved, as this information can be inadvertently revealed during inference.

### Terminology and Concepts
- **Canaries**: Canaries are carefully inserted synthetic sequences within the training data, designed to test if the model will memorize specific content. In a real-world example, a canary might be a phrase that mimics sensitive information, like a fake social security number or password. By inserting canaries and then querying the model, researchers can observe if these sequences are memorized and retrieved, indicating potential privacy risks.

   For the use case, different patterns of canaries were generated by adjusting the pattern structure, the repetition scheme, and the vocabulary used. Here are some common pattern types:

   1. **Fixed Pattern with Variable Vocabulary**
      - *Pattern*: `{fruit1} loves {fruit2}`
      - *Example Canaries*: “apple loves orange,” “grape loves banana”
   
   2. **Multi-Repetition Patterns**
      - *Pattern*: Repetition of phrases or keywords.
      - *Example Canaries*: “error code 404” repeated five times, “urgent alert” repeated twice
   
   3. **Sequential Patterns**
      - *Pattern*: `{number1}, {number2}, {number3}`
      - *Example Canaries*: “1, 2, 3,” “alpha, beta, gamma”
   
   4. **Hybrid Patterns**
      - *Pattern*: `{adjective} {noun} {verb}`
      - *Example Canaries*: “quick fox jumps,” “lazy dog sleeps”
   
   5. **Sensitive Data Pattern**
      - *Pattern*: Sensitive-style identifiers like “security PIN” or “credit card” phrases to test privacy responses.
      - *Example Canaries*:
        - “My security PIN is 123-456-789”
        - “Please use code 987-654-321 to access”
        - “Bank verification code: 1122-3344-5566”

- **Exposure Metrics**: The exposure metric quantifies how likely a model will memorize a sequence. Specifically, the exposure of a canary is calculated by examining its rank relative to other sequences, determined by the log-perplexity score. A lower log-perplexity implies a higher likelihood of memorization, indicating that the model can retrieve or reconstruct the canary when prompted with a related query.

   Given a canary \( s[r] \), a model with parameters \( q \), and the randomness space \( R \), the exposure of \( s[r] \) is defined as:

  **exposure<sub>q</sub>(s[r]) = log<sub>2</sub> |R| - log<sub>2</sub> rank<sub>q</sub>(s[r])**

  where \( |R| \) is a constant. Thus, the exposure is essentially computing the negative log-rank, with an added constant to ensure the exposure value is always positive.

  The exposure is a real value ranging between 0 and \( log<sub>2</sub> |R| \). Its maximum can be achieved only by the most likely, top-ranked canary, and conversely, its minimum value of 0 corresponds to the least likely canary.

  Exposure is not a normalized metric; the magnitude of exposure values depends on the search space size. This characteristic of exposure values highlights that revealing a unique secret from a vast search space can be highly damaging, whereas guessing one out of a small, easily enumerated set of secrets may be less concerning.

## Testing Methodology for Memorization
The testing methodology introduced in *The Secret Sharer* involves calculating the exposure of inserted canaries in trained models. This process entails:

1. **Inserting Canaries**: Randomly generated sequences are embedded within the training data, mimicking potentially sensitive data.
2. **Training the Model**: A generative model, such as GPT-2, is trained on the dataset containing these canaries.
3. **Calculating Exposure**: The model is then queried to see if it can reconstruct the canaries, and the exposure metric is calculated to assess memorization.

## Installation
To get started you can: **Clone the Repository:**
   ```bash
   git clone https://github.com/Vanthoff007/SecretSharer.git
   ```

---

---
# Generate Canaries

## 1. `CanaryDatasetGenerator`
This class generates a dataset of unique canary sequences and reference sequences with specified patterns and repetitions.

### Arguments
- **`vocabulary`** *(list of str)*: List of tokens that can be used to generate sequences.
- **`pattern`** *(str)*: Pattern string with placeholders `{}` for formatting each canary sequence.
- **`repetitions`** *(list of int)*: List specifying the number of times each canary set should be repeated.
- **`secrets_per_repetition`** *(list of int)*: List specifying the number of unique canary sequences in each repetition group.
- **`num_references`** *(int)*: Number of reference sequences to be generated.
- **`seed`** *(int, optional)*: Seed value for random number generation for reproducibility. Default is `0`.

### Methods
- **`create_dataset()`**: Generates a dataset of canary sequences with repetitions and reference sequences.
  - **Returns**: `dict` containing:
    - **`dataset`** *(list of str)*: List of canary sequences, with specified repetitions.
    - **`references`** *(list of str)*: List of unique reference sequences.

---

# Compute Perplexity

## 2. `PerplexityCalculator`
This class computes perplexity scores for a set of sequences (canaries and references) using a language model.

### Arguments
- **`model`**: A pre-trained language model (e.g., `transformers` model). Default is `None`.
- **`tokenizer`**: A tokenizer compatible with the pre-trained model, used to tokenize input texts. Default is `None`.
- **`max_length`** *(int, optional)*: Maximum token length for each input sequence. Default is `1024`.
- **`device`** *(str or `torch.device`, optional)*: Device for model inference, either `'cpu'` or `'cuda'`. Default is `torch.device("cpu")`.

### Methods
- **`compute_perplexity(text)`**: Computes perplexity for a single text sequence.
  - **Arguments**:
    - **`text`** *(str)*: The text sequence for which to compute perplexity.
  - **Returns**: `float`, representing the computed perplexity score.

- **`compute_perplexities_for_canaries(canaries, references)`**: Computes perplexity scores for a list of canary and reference sequences.
  - **Arguments**:
    - **`canaries`** *(list of str)*: List of unique canary text sequences.
    - **`references`** *(list of str)*: List of reference text sequences.
  - **Returns**: `tuple` containing:
    - **`canary_perplexities`** *(list of float)*: Perplexity scores for each canary sequence.
    - **`reference_perplexities`** *(list of float)*: Perplexity scores for each reference sequence.

---

# Compute Exposure

## 3. `ComputeExposure`
This class calculates exposure scores for a set of canaries based on their perplexities relative to reference perplexities.

### Arguments
- **`perplexities`** *(dict, optional)*: Dictionary where keys are canary identifiers and values are their perplexity values. Default is an empty dictionary `{}`.
- **`reference_perplexities`** *(list, optional)*: List of perplexity values for reference sequences. Default is an empty list `[]`.

### Methods
- **`compute_exposure_rank_method()`**: Computes exposure scores based on the perplexity ranks.
  - **Returns**: `dict` where keys are canary identifiers, and values are the exposure scores (as `float`).

---
