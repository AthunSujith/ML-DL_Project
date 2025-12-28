# Tutorial by www.pylessons.com
# Tutorial written for - Tensorflow 2.3.1

# Import required libraries
import os                      # For handling file paths and operations
import random                  # For random number generation
import gym                     # OpenAI Gym for reinforcement learning environments
import pylab                   # For plotting graphs
import numpy as np            # For numerical operations
from collections import deque  # For creating memory buffer
import tensorflow as tf        # Main deep learning framework
from tensorflow.keras.models import Model, load_model  # For creating and loading models
from tensorflow.keras.layers import Input, Dense       # Neural network layers
from tensorflow.keras.optimizers import Adam, RMSprop  # Optimization algorithms


def OurModel(input_shape, action_space):
    # Create input layer
    X_input = Input(input_shape)
    X = X_input

    # Create neural network architecture
    # First hidden layer with 512 neurons
    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X)

    # Second hidden layer with 256 neurons
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    
    # Third hidden layer with 64 neurons
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    # Output layer with neurons equal to action space
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    # Create and compile the model
    model = Model(inputs = X_input, outputs = X)
    model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    model.summary()
    return model

class DQNAgent:
    def __init__(self, env_name):
        # Initialize environment
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.env.seed(0)  
        self.env._max_episode_steps = 4000  # Set maximum steps per episode
        self.state_size = self.env.observation_space.shape[0]  # Get state space size
        self.action_size = self.env.action_space.n  # Get action space size

        # Training parameters
        self.EPISODES = 1000  # Total number of training episodes
        self.memory = deque(maxlen=2000)  # Memory buffer for experience replay
        
        # Agent parameters
        self.gamma = 0.95    # Discount factor for future rewards
        self.epsilon = 1.0   # Initial exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.999  # Decay rate for exploration
        self.batch_size = 32  # Batch size for training
        self.train_start = 1000  # Start training after this many memories

        # Model parameters
        self.ddqn = True  # Use Double DQN if True
        self.Soft_Update = False  # Use soft update if True
        self.TAU = 0.1  # Soft update parameter

        # Setup for saving models and plotting
        self.Save_Path = 'Models'
        self.scores, self.episodes, self.average = [], [], []
        
        # Set model name based on type
        if self.ddqn:
            print("----------Double DQN--------")
            self.Model_name = os.path.join(self.Save_Path,"DDQN_"+self.env_name+".h5")
        else:
            print("-------------DQN------------")
            self.Model_name = os.path.join(self.Save_Path,"DQN_"+self.env_name+".h5")
        
        # Create main and target networks
        self.model = OurModel(input_shape=(self.state_size,), action_space = self.action_size)
        self.target_model = OurModel(input_shape=(self.state_size,), action_space = self.action_size)

    def update_target_model(self):
        # Update target network weights
        if not self.Soft_Update and self.ddqn:
            # Hard update - copy weights directly
            self.target_model.set_weights(self.model.get_weights())
            return
        if self.Soft_Update and self.ddqn:
            # Soft update - gradually update weights
            q_model_theta = self.model.get_weights()
            target_model_theta = self.target_model.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1-self.TAU) + q_weight * self.TAU
                target_model_theta[counter] = target_weight
                counter += 1
            self.target_model.set_weights(target_model_theta)

    def remember(self, state, action, reward, next_state, done):
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, done))
        # Decay epsilon if enough samples are gathered
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        # Choose action using epsilon-greedy policy
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        else:
            return np.argmax(self.model.predict(state))  # Exploit

    def replay(self):
        # Experience replay method
        if len(self.memory) < self.train_start:
            return
        
        # Sample random minibatch from memory
        minibatch = random.sample(self.memory, min(self.batch_size, self.batch_size))

        # Initialize arrays for batch processing
        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # Fill arrays with sampled experiences
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # Batch prediction for efficiency
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)
        target_val = self.target_model.predict(next_state)

        # Update target values based on DDQN or DQN algorithm
        for i in range(len(minibatch)):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                if self.ddqn:
                    a = np.argmax(target_next[i])
                    target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])   
                else:
                    target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the network
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def load(self, name):
        # Load model weights
        self.model = load_model(name)

    def save(self, name):
        # Save model weights
        self.model.save(name)

    pylab.figure(figsize=(18, 9))
    def PlotModel(self, score, episode):
        # Plot training progress
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores) / len(self.scores))
        pylab.plot(self.episodes, self.average, 'r')
        pylab.plot(self.episodes, self.scores, 'b')
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Steps', fontsize=18)
        dqn = 'DQN_'
        softupdate = ''
        if self.ddqn:
            dqn = 'DDQN_'
        if self.Soft_Update:
            softupdate = '_soft'
        try:
            pylab.savefig(dqn+self.env_name+softupdate+".png")
        except OSError:
            pass

        return str(self.average[-1])[:5]
    
    def run(self):
        # Main training loop
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                #self.env.render()
                # Select and perform action
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                
                # Adjust reward based on termination condition
                if not done or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                    
                # Store experience and learn
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:
                    self.update_target_model()
                    average = self.PlotModel(i, e)
                    print("episode: {}/{}, score: {}, e: {:.2}, average: {}".format(e, self.EPISODES, i, self.epsilon, average))
                    if i == self.env._max_episode_steps:
                        print("Saving trained model as cartpole-ddqn.h5")
                        break
                self.replay()

    def test(self):
        # Test the trained agent
        self.load("cartpole-ddqn.h5")
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                    break

if __name__ == "__main__":
    env_name = 'CartPole-v1'
    agent = DQNAgent(env_name)
    agent.run()
    #agent.test()
