# HW6-HMM

In this assignment, you'll implement the Forward and Viterbi Algorithms (dynamic programming). 


# Assignment

## Overview 

The goal of this assignment is to implement the Forward and Viterbi Algorithms for Hidden Markov Models (HMMs). These algorithms are fundamental for determining the probability of an observed sequence (Forward Algorithm) and the most likely sequence of hidden states given the observed data (Viterbi Algorithm).

For a helpful refresher on HMMs and the Forward and Viterbi Algorithms you can check out the resources [here](https://web.stanford.edu/~jurafsky/slp3/A.pdf), 
[here](https://towardsdatascience.com/markov-and-hidden-markov-model-3eec42298d75), and [here](https://pieriantraining.com/viterbi-algorithm-implementation-in-python-a-practical-guide/). 

---

## Installation

### Local Installation (with Flit)

1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/y-umay/HW6-HMM.git
   cd HW6-HMM
   \`\`\`

2. Install the package:
   \`\`\`bash
   pip install flit
   flit install
   \`\`\`

After installation, you can import the model:
\`\`\`python
from hmm import HiddenMarkovModel
\`\`\`

---

## Running the Tests

To run the tests with detailed output:
\`\`\`bash
python -m pytest -vv test/*py
\`\`\`

Or using the basic command:
\`\`\`bash
pytest test_hmm.py
\`\`\`

Tests cover both the mini and full weather datasets, including edge cases.

---

## GitHub Actions (CI)

Automated testing runs on every push or pull request.  
View the latest build status in the [Actions tab](https://github.com/y-umay/HW6-HMM/actions).

---

## Methods

### HiddenMarkovModel Class
The `HiddenMarkovModel` class encapsulates the following components:
- **Initialization (`__init__`)**: Loads the observation states, hidden states, prior probabilities, transition probabilities, and emission probabilities into the model.
- **Forward Algorithm (`forward`)**: 
  - Calculates the probability (likelihood) of an observed sequence.
  - Uses dynamic programming with an alpha matrix for probabilities.
  - Edge cases handled: empty sequences, single observations, and zero-probability transitions.
- **Viterbi Algorithm (`viterbi`)**: 
  - Finds the most likely sequence of hidden states for a given observed sequence.
  - Employs dynamic programming with a viterbi table and backpointer matrix.
  - Handles edge cases similar to the forward algorithm.

---

## Data 
The assignment includes two datasets:
1. **mini_weather_hmm.npz**: For initial testing and debugging.
2. **full_weather_hmm.npz**: For final evaluation.

Each dataset contains:
- `hidden_states`: Possible hidden states.
- `observation_states`: Possible observable states.
- `prior_p`: Prior probabilities for hidden states.
- `transition_p`: Transition probabilities between hidden states.
- `emission_p`: Emission probabilities from hidden states to observed states.

Additionally, observation sequences and their corresponding best hidden state sequences are provided:
- `observation_state_sequence`
- `best_hidden_state_sequence`

---

## Unit Testing

### Tests Implemented
Tests are implemented using `pytest` in `test_hmm.py`.

### Mini Weather Model
- **Correctness Tests**:
  - Verifies that the forward algorithm returns a positive probability.
  - Confirms that the Viterbi sequence matches the expected hidden states.
- **Edge Cases**:
  - Empty observation sequence (raises `IndexError` or `ValueError`).
  - Single observation sequence returns valid results.
  - Zero-probability transition matrix results in zero forward probability.

### Full Weather Model
- **Correctness Tests**:
  - Forward algorithm returns a positive probability.
  - Viterbi algorithm returns the correct sequence and correct sequence length.

---

## Edge Cases Handled
1. **Empty Observation Sequence**: Properly raises exceptions.
2. **Single Observation**: Returns valid probabilities and sequences.
3. **Zero-Probability Transitions**: Forward probability returns zero; Viterbi handles gracefully.

---

## Usage Example

\`\`\`python
from hmm import HiddenMarkovModel
import numpy as np

hmm_model = HiddenMarkovModel(
    hidden_states=["Rainy", "Sunny"],
    observation_states=["Walk", "Shop", "Clean"],
    prior_p=np.array([0.6, 0.4]),
    transition_p=np.array([[0.7, 0.3], [0.4, 0.6]]),
    emission_p=np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
)

observations = ["Walk", "Shop", "Clean"]
viterbi_path = hmm_model.viterbi(observations)
forward_prob = hmm_model.forward(observations)

print("Best hidden state sequence:", viterbi_path)
print("Forward probability:", forward_prob)
\`\`\`

---

## Task List

- [x] Complete the `HiddenMarkovModel` class methods
  - [x] Implement `forward` function
  - [x] Implement `viterbi` function
- [x] Unit Testing
  - [x] Mini weather dataset
  - [x] Full weather dataset
  - [x] Edge cases (empty input, single observation, zero-probability transitions)
- [x] Updated README with methods description
- [x] Optional: Make the module pip installable
- [x] Optional: Set up GitHub Actions for automated testing

---

## License

This project is licensed under the MIT License. See \`LICENSE\` for details.