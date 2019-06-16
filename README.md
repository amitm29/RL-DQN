# Reinforcement Learning on Games

It uses a DQN Agent (Deep Q-Learner) to play different games.

Agent class defined in DQN_Agent.py models a DQN Agent.
This Agent class is used to play different games. 
<a href = "https://gym.openai.com/">OpenAI gym</a> module is used to create environments for these games.

It can be used to play:-
1. <a href = "https://gym.openai.com/envs/CartPole-v1/">CartPole game</a>
2. <a href = "https://gym.openai.com/envs/MountainCar-v0/">MountainCar game</a>
3. <a href = "https://gym.openai.com/envs/Acrobot-v1/">Acrobot game</a>

It can also be used to play other games provided by the gym module, by adding a few lines of code in Agent class present in DQN_Agent.py.

## Installing gym
To install <a href = "https://gym.openai.com/">OpenAI gym</a> module you need to run the folliwing command into the terminal:-

```
pip install gym
```

## Playing games using DQN Agent
The repository contains trained models for DQN agents for the 3 games listed above. You can either use these trained models directly or you can also train your own.

#### Playing games using trained model of DQN Agent
To use the trained model of the DQN agent you need to clone this repository by writing the following command into the terminal-

```
git clone https://github.com/amitm29/Reinforcement-Learning-on-Games.git
```

After cloning the repository, you need to open the RL bot.ipynb jupyter notebook, and run all the cells except the one for training the agent:-

```python
agent1.train_agent(n_episodes=1100)
```

When you run the cell:-
```python
agent1.play(5)
```
The agent will first load the best model previously trained and then start playing 5 episodes of the game using that model.


### Playing games by training your own DQN Agent
To train your own DQN Agent you need to follow the following steps:-

1. Create your agent

   ```python
   agent = Agent([24,12,12],game='CartPole-v0')
   ```
   Agent constructor takes the follwing parameters:-
   
        __init__(self,brain_layers=[12,24],state_size=None,action_size=None,max_steps=400,game='MountainCar-v0')
        
        brain_layers :  list of neurons in the hidden layers of the NN of the brain of the Agent.
        state_size   :  number of variables representing the state of the environment.
                       if None, then automatically gets the value depending upon the environment.
        action_size  :  total number of possible actions that the agent can take.
                       if None, then automatically gets the value depending upon the environment.
        max_steps    :  maximum allowed steps in an episode of the game.
        game         :  name of the game environment to be loaded from the gym module.
 
 
2. Train your agent

   ```python
   agent.train_agent(n_episodes=1100)
   ```
   agent.train_agent() trains the agent by playing **n_episodes** of the game, and using reinforcement learning, it takes the following parameters:-
   
        def train_agent(self,n_episodes=1000,render=False,batch_size=64,load_prev_best=False)
        
        n_episodes     :  no. of episodes to be played.
        render         :  whether to render and display the game or not.
        batch_size     :  no. of samples to be taken from agent's memory to train the NN after each episode.
        load_prev_best :  True : loads the previous best model saved .
                          False : Does nothing.
                          
        NOTE : Setting render=True will slow down the training process.
        NOTE : You can produce KeyboardInterrupt to stop training the model at any point in between 
               and safely close the environment.


3. Playing the game using your agent

   ```python
   agent.play(5)
   ```
   agent.play() enables the agent to play the game using its knowledge, it takes the following parameters:-
   
        n_episodes  :  no. of episodes to be played.
        load_model  :  True  : loads the best model saved while training 
                               or from model_path (if provided).
                       False : uses the current weights model only.
        max_steps   :  no. of steps in each episode.
                       <=self.max_steps : uses self.max_steps
                       >self.max_steps  : uses max_steps
        model_path  :  None : best model is loaded automatically.
                       else : loads the model from model_path if it exists,
                              if it doesn't, loads the best model only.
        
