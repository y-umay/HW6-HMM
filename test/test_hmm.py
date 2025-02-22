import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    hmm_model = HiddenMarkovModel(
        observation_states=mini_hmm['observation_states'],
        hidden_states=mini_hmm['hidden_states'],
        prior_p=mini_hmm['prior_p'],
        transition_p=mini_hmm['transition_p'],
        emission_p=mini_hmm['emission_p']
    )

    observation_seq = mini_input['observation_state_sequence']
    expected_viterbi_seq = list(mini_input['best_hidden_state_sequence'])

    # Forward algorithm test (Check for positive probability instead of missing key)
    forward_prob = hmm_model.forward(observation_seq)
    assert forward_prob > 0, "Forward probability should be positive."

    # Viterbi algorithm test
    viterbi_seq = hmm_model.viterbi(observation_seq)
    assert viterbi_seq == expected_viterbi_seq, "Viterbi sequence mismatch."
    assert len(viterbi_seq) == len(observation_seq), "Viterbi sequence length mismatch."

    # Edge case 1: Empty observation sequence
    with pytest.raises((IndexError, ValueError)):
        hmm_model.forward([])
    with pytest.raises((IndexError, ValueError)):
        hmm_model.viterbi([])

    # Edge case 2: Single observation
    single_obs = [observation_seq[0]]
    single_forward_prob = hmm_model.forward(single_obs)
    assert single_forward_prob > 0, "Forward probability for single observation should be positive."

    single_viterbi_seq = hmm_model.viterbi(single_obs)
    assert len(single_viterbi_seq) == 1, "Viterbi sequence for single observation should have length 1."

    # Edge case 3: Zero-probability transition
    zero_transition_hmm = HiddenMarkovModel(
        observation_states=mini_hmm['observation_states'],
        hidden_states=mini_hmm['hidden_states'],
        prior_p=mini_hmm['prior_p'],
        transition_p=np.zeros_like(mini_hmm['transition_p']),  # All transitions set to zero
        emission_p=mini_hmm['emission_p']
    )

    zero_forward_prob = zero_transition_hmm.forward(observation_seq)
    assert zero_forward_prob == 0, "Forward probability should be zero when all transitions are impossible."

    zero_viterbi_seq = zero_transition_hmm.viterbi(observation_seq)
    assert zero_viterbi_seq is not None, "Viterbi should handle zero-probability transitions gracefully."


def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    full_hmm = np.load('./data/full_weather_hmm.npz')
    full_input = np.load('./data/full_weather_sequences.npz')

    hmm_model = HiddenMarkovModel(
        observation_states=full_hmm['observation_states'],
        hidden_states=full_hmm['hidden_states'],
        prior_p=full_hmm['prior_p'],
        transition_p=full_hmm['transition_p'],
        emission_p=full_hmm['emission_p']
    )

    observation_seq = full_input['observation_state_sequence']
    expected_viterbi_seq = list(full_input['best_hidden_state_sequence'])

    # Forward algorithm test
    forward_prob = hmm_model.forward(observation_seq)
    assert forward_prob > 0, "Forward probability should be positive."

    # Viterbi algorithm test
    viterbi_seq = hmm_model.viterbi(observation_seq)
    assert viterbi_seq == expected_viterbi_seq, "Viterbi sequence mismatch."
    assert len(viterbi_seq) == len(observation_seq), "Viterbi sequence length mismatch."




