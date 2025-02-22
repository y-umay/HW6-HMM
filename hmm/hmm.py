import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        # Step 1. Initialize variables
        num_states = len(self.hidden_states)
        num_observations = len(input_observation_states)
        
        # Initialize the forward probability matrix (alpha)
        alpha = np.zeros((num_states, num_observations))

        # Step 2. Calculate probabilities
        # Initialization step (t=0)
        for state_index in range(num_states):
            obs_index = self.observation_states_dict[input_observation_states[0]]
            alpha[state_index, 0] = self.prior_p[state_index] * self.emission_p[state_index, obs_index]

        # Recursion step (t > 0)
        for t in range(1, num_observations):
            for curr_state in range(num_states):
                obs_index = self.observation_states_dict[input_observation_states[t]]
                alpha[curr_state, t] = sum(
                    alpha[prev_state, t - 1] * self.transition_p[prev_state, curr_state]
                    for prev_state in range(num_states)
                ) * self.emission_p[curr_state, obs_index]

        # Step 3. Return final probability 
        forward_probability = np.sum(alpha[:, -1])
        return forward_probability
    

    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        # Step 1. Initialize variables
        num_states = len(self.hidden_states)
        num_observations = len(decode_observation_states)

        # Store probabilities of hidden states at each step
        viterbi_table = np.zeros((num_observations, num_states))

        # Store best path for traceback
        backpointer = np.zeros((num_observations, num_states), dtype=int)
        
        #store probabilities of hidden state at each step 
        #viterbi_table = np.zeros((len(decode_observation_states), len(self.hidden_states)))
        #store best path for traceback
        #best_path = np.zeros(len(decode_observation_states))         
        
       
       # Step 2. Calculate Probabilities
        # Initialization step (t=0)
        for state_index in range(num_states):
            obs_index = self.observation_states_dict[decode_observation_states[0]]
            viterbi_table[0, state_index] = self.prior_p[state_index] * self.emission_p[state_index, obs_index]
            backpointer[0, state_index] = 0

        # Recursion step (t > 0)
        for t in range(1, num_observations):
            for curr_state in range(num_states):
                obs_index = self.observation_states_dict[decode_observation_states[t]]

                # Calculate probability for each possible previous state
                transition_probs = [
                    viterbi_table[t - 1, prev_state] * self.transition_p[prev_state, curr_state]
                    for prev_state in range(num_states)
                ]

                # Choose the best previous state with the highest probability
                best_prev_state = np.argmax(transition_probs)
                best_transition_prob = transition_probs[best_prev_state]

                # Update the viterbi table and backpointer
                viterbi_table[t, curr_state] = best_transition_prob * self.emission_p[curr_state, obs_index]
                backpointer[t, curr_state] = best_prev_state

            
        # Step 3. Traceback 
        best_last_state = np.argmax(viterbi_table[-1, :])  # best state at the last observation
        best_path_indices = [best_last_state]

        # Traceback the path using the backpointer
        for t in range(num_observations - 1, 0, -1):
            best_last_state = backpointer[t, best_last_state]
            best_path_indices.insert(0, best_last_state)

        # Step 4. Return best hidden state sequence 
        best_hidden_state_sequence = [self.hidden_states_dict[state_index] for state_index in best_path_indices]

        return best_hidden_state_sequence
     