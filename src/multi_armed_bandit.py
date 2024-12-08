import numpy as np
import src.random


class MultiArmedBandit:
    """
    MultiArmedBandit reinforcement learning agent.

    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action.
    """

    def __init__(self, epsilon=0.2):
        self.epsilon = epsilon

    def fit(self, env, steps=1000, num_bins=100):
        """
        Trains the MultiArmedBandit on an OpenAI Gymnasium environment.

        See page 32 of Sutton and Barto's book Reinformcement Learning for
        pseudocode (http://incompleteideas.net/book/RLbook2020.pdf).
        Initialize your parameters as all zeros. For the step size, use
        1/N, where N is the number of times the current action has been
        performed. (This is the version of Bandits we saw in lecture before
        we introduced alpha). Use an epsilon-greedy approach to pick actions.

        See (https://gymnasium.farama.org/) for examples of how to use the OpenAI
        Gymnasium Environment interface.

        In every step of the fit() function, you should sample
            two random numbers using functions from `src.random`.
            1.  First, use either `src.random.rand()` or `src.random.uniform()`
                to decide whether to explore or exploit.
            2. Then, use `src.random.choice` or `src.random.randint` to decide
                which action to choose. Even when exploiting, you should make a
                call to `src.random` to break (possible) ties.

        Please don't use `np.random` functions; use the ones from `src.random`!
        Please do not use `env.action_space.sample()`!

        Hints:
          - Use env.action_space.n and env.observation_space.n to get the
            number of available actions and states, respectively.
          - Remember to reset your environment at the end of each episode. To
            do this, call env.reset() whenever the value of "terminated or truncated" returned
            from env.step() is True.
          - When choosing to exploit the best action rather than exploring,
            do not use np.argmax: it will deterministically break ties by
            choosing the lowest index of among the tied values. Instead,
            please *randomly choose* one of those tied-for-the-largest values.
          - MultiArmedBandit treats all environment states the same. However,
            in order to have the same API as agents that model state, you must
            explicitly return the state-action-values Q(s, a). To do so, just
            copy the action values learned by MultiArmedBandit S times, where
            S is the number of states.

        Arguments:
          env - (Env) An OpenAI Gymnasium environment with discrete actions and
            observations. See the OpenAI Gymnasium documentation for example use
            cases (https://gymnasium.farama.org/api/env/).
          steps - (int) The number of actions to perform within the environment
            during training.

        Returns:
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.
          rewards - (np.array) A 1D sequence of averaged rewards of length `num_bins`.
            Let s = int(np.ceil(steps / `num_bins`)), then rewards[0] should
            contain the average reward over the first s steps, rewards[1]
            should contain the average reward over the next s steps, etc.
            Please note that: The total number of steps will not always divide evenly by the 
            number of bins. This means the last group of steps may be smaller than the rest of 
            the groups. In this case, we can't divide by s to find the average reward per step 
            because we have less than s steps remaining for the last group.
        """
        n_actions = env.action_space.n
        n_states = env.observation_space.n

        # Initialize Q-values and action counts
        self.Q = np.zeros(n_actions)
        self.N = np.zeros(n_actions)

        # Initialize reward tracking
        avg_rewards = np.zeros(num_bins)
        all_rewards = []

        # Reset environment
        current_state, _ = env.reset()

        # Number of steps per bin
        bin_size = int(np.ceil(steps / num_bins))

        # Loop over total steps
        for step in range(steps):
            # Epsilon-greedy policy for action selection
            if src.random.rand() < self.epsilon:
                # Explore: Random action
                action = src.random.randint(0, n_actions)
            else:
                # Exploit: Choose action with the highest Q-value (random tie-breaking)
                max_value = np.max(self.Q)
                best_actions = [a for a in range(n_actions) if self.Q[a] == max_value]
                action = src.random.choice(best_actions)

            # Perform action in the environment
            _, reward, terminated, truncated, _ = env.step(action)
            all_rewards.append(reward)

            # Update Q-values using incremental formula
            self.N[action] += 1
            self.Q[action] += (1 / self.N[action]) * (reward - self.Q[action])

            # If the episode ends, reset the environment
            if terminated or truncated:
                current_state, _ = env.reset()

            # Compute bin rewards dynamically
            if (step + 1) % bin_size == 0 or step == steps - 1:
                bin_index = (step + 1) // bin_size - 1 if (step + 1) % bin_size == 0 else num_bins - 1
                start = bin_index * bin_size
                end = min((bin_index + 1) * bin_size, steps)
                avg_rewards[bin_index] = np.mean(all_rewards[start:end])

        # Copy Q-values across all states to create state-action values
        state_action_values = np.tile(self.Q, (n_states, 1))

        return state_action_values, avg_rewards

    def predict(self, env, state_action_values):
        """
        Runs prediction on an OpenAI environment using the policy defined by
        the MultiArmedBandit algorithm and the state action values. Predictions
        are run for exactly one episode. Note that one episode may produce a
        variable number of steps.

        Hints:
          - You should not update the state_action_values during prediction.
          - Exploration is only used in training. During prediction, you
            should only "exploit."
          - You should use a loop to predict over each step in an episode until
            it terminates by returning `terminated or truncated=True`.
          - When choosing to exploit the best action, do not use np.argmax: it
            will deterministically break ties by choosing the lowest index of
            among the tied values. Instead, please *randomly choose* one of
            those tied-for-the-largest values.

        Arguments:
          env - (Env) An OpenAI Gymnasium environment with discrete actions and
            observations. See the OpenAI Gymnasium documentation for example use
            cases (https://gymnasium.farama.org/api/env/).
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.

        Returns:
          states - (np.array) The sequence of states visited by the agent over
            the course of the episode. Does not include the starting state.
            Should be of length K, where K is the number of steps taken within
            the episode.
          actions - (np.array) The sequence of actions taken by the agent over
            the course of the episode. Should be of length K, where K is the
            number of steps taken within the episode.
          rewards - (np.array) The sequence of rewards received by the agent
            over the course  of the episode. Should be of length K, where K is
            the number of steps taken within the episode.
        """
        state, _ = env.reset()

        states = []
        actions = []
        rewards = []

        terminated, truncated = False, False
        while not (terminated or truncated):
            # Exploit: Choose the best action (random tie-breaking)
            max_value = np.max(state_action_values[state])
            best_actions = [a for a in range(len(state_action_values[state])) if state_action_values[state][a] == max_value]
            action = src.random.choice(best_actions)

            # Take the action
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Record the results
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            # Move to the next state
            state = next_state

        return np.array(states), np.array(actions), np.array(rewards)