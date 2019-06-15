# !pip install gym
from collections import deque
import gym
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import os


class Agent:
    def __init__(self,brain_layers=[12,24],state_size=None,action_size=None,
                 max_steps=400,game='MountainCar-v0'):
        '''
        brain_layers : list of neurons in the hidden layers of the NN of the brain of the Agent
        state_size : number of variables representing the state of the environment
                     if None, then automatically gets the value depending upon the environment.
        action_size : total number of possible actions that the agent can take
                      if None, then automatically gets the value depending upon the environment.
        max_steps : maximum allowed steps in an episode of the game
        game : name of the game environment to be loaded from the gym module
        '''
        self.game = game
        self.env = gym.make(game)
        self.env._max_episode_steps = max_steps
        self.max_steps = max_steps
        self.state_size = self.env.observation_space.shape[0] if state_size==None else state_size
        self.action_size = self.env.action_space.n if action_size==None else action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95   #discount factor
        # Exploratioln vs Exploitation tradeoff
        #Exploration : good in the begining because it helps the agent to try out different random things.
        #Exploitation : Sample good experiences from the past(memory) --> Good in the end of the training.
        #(eq.after the agent has played 100 games, it has done enough exploration and can now exploit its knowledge about the environment)
        
        self.epsilon=1.0  #the agent is going to do 100% random exploration in the begining
        self.epsilon_decay=0.995  #factor by which epsilon will decrease after each episode
        self.epsilon_min=0.1  #minimum value of epsilon, i.e. after the agent has learned enough then also we are allowing some random exploration.
        self.learning_rate=0.001  #learning rate of the NN(brain of the agent)
        self.model = self.create_model(brain_layers) #model is a NN(brain of the agent)
        
    def describe(self):
        '''
        describes the agent
        '''
        print('Game                    :       '+self.game)
        print('Max Steps               :       '+str(self.max_steps))
        print('State Size              :       '+str(self.state_size))
        print('Action Size             :       '+str(self.action_size))
        print('Discount Factor (gamma) :       '+str(self.gamma))
        print('Epsilon                 :       '+str(self.epsilon))
        print('Epsilon Decay           :       '+str(self.epsilon_decay))
        print('Epsilon Min             :       '+str(self.epsilon_min))
        print('Learning Rate           :       '+str(self.learning_rate))
        print("\n\n Agent's Brain:-")
        self.model.summary()
    
    def create_model(self, layers):
        '''
        creates a Neural Net (brain of the agent)
        Input Layer : contains neurons equal to no. of states (state_size)
        Hidden Layers : contains neurons as described by "layers"
        Output Layer : contains neurons equal to no. of possible actions (action_size)
        '''
        l = len(layers)
        model = Sequential()
        model.add(Dense(layers[0],input_dim=self.state_size,activation='relu'))
        
        for i in range(1,l):
            model.add(Dense(layers[i],activation='relu'))
            
        model.add(Dense(self.action_size,activation='linear'))
        
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self,state,action,reward,next_state,done):
        '''
        enables the agent to remember past experiences.
        it stores the past experiences of the agent in a deque, with a fixed max length.
        
        state : current state of the environment.
        action : action taken by the agent.
        reward : reward received by the agent for the action it has perrformed.
        next_state : state of the environment after the action is performed.
        done : if True, current episode is over
        '''
        self.memory.append((state,action,reward,next_state,done))
    
    def act(self,state):
        '''
        enables the agent to take an action based upon the value of epsilon:-
        the agent either takes a random action or it uses its brain (the NN) to take an action depending
        upon the current state, future possible states and the reward it may receive from them 
        
        state : current state of the environment
        '''
        #Sampling according to the greedy epsilion method
        if np.random.rand()<=self.epsilon:
            #Take a random action
            return random.randrange(self.action_size) #return 0 or 1, if actionsize=2
        else:
            #Ask neural network to give the suitable action
            return np.argmax(self.model.predict(state)[0])
    
    def train(self,batch_size=32):
        '''
        trains the NN (the brain of the agent) by using a technique called "Replay Buffer":-
        
        first it samples a batch_size no. of samples from the agent's memory and then 
        by using Stochastic Gradient Descent(SGD), it passes these samples one by one 
        through the NN and update its weights.
        
        batch_size : no. of samples to be taken from the Agent's memory for traing its brain
        '''
        #training using the replay buffer
        minibatch = random.sample(self.memory,batch_size)
        for experience in minibatch:
            state,action,reward,next_state,done = experience
            
            #we need x and y to train our neural network
            #X : state
            #Y : expected reward
            
            if not done:
                #game is not over
                #we will use bellman equation to approx the target_val of reward
                #BELLMAN EQUATION;-
                #reward = immediate_reward + max_reward_possible_from_the_future_states 
                target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])
            else:
                target = reward
            
            target_f = self.model.predict(state)
            target_f[0][action] = target
            
            #x = state, Y = target_f
            self.model.fit(state,target_f,epochs=1,verbose=0)
            
        if self.epsilon>self.epsilon_min:
            self.epsilon*=self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
                
    def load(self,name):
        '''
        loads previously saved model to the NN
        '''
        try:
            self.model.load_weights(name)
            print(name+' loaded successfully.')
            return True
        except OSError:
            print("OSError : can't load "+name)
            return False
        
    def save(self,name):
        '''
        saves the current model
        '''
        self.model.save_weights(name)
    
    def reward(self,*args):
        '''
        calculates the reward for each action taken by the agent
        called by train_agent() at each step
        '''
        if self.game=='CartPole-v0':
            #--CartPole-v0
            reward = args[1] if not args[0] else +20 if args[1]>300 else +10 if args[1]>200 else -20 if args[1]<120 and args[2]>600 else -10
        elif self.game=='MountainCar-v0':
            #--MountainCar-v0
            pos = args[0]
            #Reward based on car position
            reward = (pos+0.6)**2
            #Reward on task completion
            if pos>=0.5:
                reward+=100
            elif pos>0.3:
                reward+=5
            elif pos>0.2:
                reward+=3
            elif pos>0.1:
                reward+=2
        elif self.game=='Acrobot-v1':
            #--Acrobot-v1
            reward=500 if args[0] and args[2]+1<self.max_steps else args[1]
        else:
            print('Reward System for this game not defined yet')
            print('Go and edit the reward function in Agent class')
            return

        return reward
    
    def train_agent(self,n_episodes=1000,render=False,batch_size=64,load_prev_best=False):
        '''
        trains the agent:-
        n_episodes : no. of episodes to be played.
        render : whether to render and display the game or not.
        NOTE : setting render=True will slow down the training process.
        batch_size : no. of samples to be taken from memory to train the NN after each episode.
        load_prev_best : True : loads the previous best model saved .
                         False : Does nothing.
                         
        NOTE:You can produce KeyboardInterrupt to stop training the model at any point in between 
        and safely close the environment.
        '''
        
        #creates a directory "models" to store the models
        if not 'models' in os.listdir():
            os.mkdir('models')
        
        print(f'Exploration Rate : {int(self.epsilon*10000)/10000}')
        
        if load_prev_best:
            x = self.load('models/best_model_'+self.game+'.hdf5')
            if not x:
                print('Model Not Found : models/best_model_'+self.game+'.hdf5')
                print('Training model from scratch...')

        max_score={'score':-100000,'episode':0,'steps':self.max_steps,'pos':0}
        
        try:
            for e in range(n_episodes):
                state=self.env.reset()
                state=np.reshape(state,[1,-1])
                total_reward=0
                
                if self.game=='MountainCar-v0':
                  #--MountainCar-v0
                    max_pos=-1.2

                for t in range(self.max_steps):
                    self.env.render() if render else None
                    action = self.act(state) 
                    next_state,reward,done,other_info = self.env.step(action)
                    next_state = np.reshape(next_state,[1,-1])
            
                    if self.game=='CartPole-v0':
                        #--CartPole-v0
                        reward = self.reward(done,reward,e)
                    elif self.game=='MountainCar-v0':
                        #--MountainCar-v0
                        pos = next_state[0][0]
                        if pos>max_pos:
                            max_pos=pos
                        reward=self.reward(pos)
                        total_reward+=reward
                    elif self.game=='Acrobot-v1':
                        #--Acrobot-v1
                        reward=self.reward(done,reward,t)
                        total_reward+=reward
                    else:
                        self.reward()
                        print('Add reward code and call to reward function')
                        return
                    
                    self.remember(state,action,reward,next_state,done)
                    state=next_state

                    if done:
                        #Game episode is over
                        if self.game=='CartPole-v0':
                            #--CartPole-v0
                            if t+1>=max_score['score']:
                                max_score['score']=t+1
                                max_score['episode']=e
                                #save the best model
                                self.save("models/best_model_"+self.game+".hdf5")
                            print(f'Game Episode {e+1}/{n_episodes} : Score {t+1}, Exploration Rate {int(self.epsilon*10000)/10000}')
                        
                        elif self.game=='MountainCar-v0':
                            #--MountainCar-v0
                            #Less the number of steps better is the score
                            if t+1<max_score['steps']:
                                max_score['score']=total_reward
                                max_score['episode']=e
                                max_score['steps']=t+1
                                max_score['pos']=max_pos
                                #save the best model
                                self.save("models/best_model_"+self.game+".hdf5")
                            print(f'Game Episode {e+1}/{n_episodes} : Score {int(total_reward*100)/100}, Max Pos {int(max_pos*100)/100}, Steps {t+1} , Exploration Rate {int(self.epsilon*10000)/10000}')
                            
                        elif self.game=='Acrobot-v1':
                            #--Acrobot-v1
                            if total_reward>=max_score['score'] and t+1<max_score['steps']:
                                max_score['score']=total_reward
                                max_score['episode']=e
                                max_score['steps']=t+1
                                #save the best model
                                self.save("models/best_model_"+self.game+".hdf5")
                            print(f'Game Episode {e+1}/{n_episodes} : Score {total_reward}, Steps {t+1} , Exploration Rate {int(self.epsilon*10000)/10000}')
                        
                        else:
                            print('Write code for what to do after an episode is over')
                            return
                        break
                
                #if memory has more samples than batch_size then train the NN
                if len(self.memory)>batch_size:
                    self.train(batch_size)
                #after every 50 episodes save the model
                if (e+1)%50==0:
                    self.save("models/weights_"+'{:04d}'.format(e+1)+'_'+self.game+".hdf5")

            self.env.close()
            print('Deep Q-Learner Model Trained')
        except KeyboardInterrupt:
            self.env.close()
        finally:
            print(max_score)


        
    def play(self,n_episodes,load_model=True,max_steps=0,model_path=None):
        '''
        enables the agent to play the game
        n_episodes : no. of episoeds to be played
        load_model : True : loads the best model saved while training 
                            or from model_path (if provided).
                     False : uses the current weights model only.
        max_steps : no. of steps in each episode
                    <=self.max_steps : uses self.max_steps
                    >self.max_steps  : uses max_steps
        model_path : None : best model is loaded automatically
                     else : loads the model from model_path if it exists,
                            if it doesn't, loads the best model only
        '''
        #sets the epsilon to its minimum value, so that the agent can exploit its knowledge
        self.epsilon=self.epsilon_min
        #sets new value to the no. of steps in an episode allowed by the environment
        self.env._max_episode_steps=max(self.max_steps,max_steps)
        #loads the best model if load_model=True 
        if load_model:
            if model_path==None:
                x = self.load('models/best_model_'+self.game+'.hdf5')
                if not x:
                    print('Model Not Found : models/best_model_'+self.game+'.hdf5')
                    print('Train your agent using train_agent()')
                    return
            else:
                x = self.load(model_path)
                if not x:
                    print('Model Not Found : '+model_path)
                    print('Loading models/best_model_'+self.game+'.hdf5 instead...')
                    x = self.load('models/best_model_'+self.game+'.hdf5')
                    if not x:
                        print('Model Not Found : models/best_model_'+self.game+'.hdf5')
                        print('Train your agent using train_agent()')
                        return
        #plays n_episodes episodes of the game
        try:
            for e in range(n_episodes):
                state=self.env.reset()
                state=np.reshape(state,[1,-1])
                total_reward=0
                if self.game=='MountainCar-v0':
                    #--MountainCar-v0
                    max_pos=-1.2
                for t in range(max(self.max_steps,max_steps)):
                    self.env.render()
                    action = self.act(state)
                    next_state,reward,done,other_info = self.env.step(action)
                    next_state = np.reshape(next_state,[1,-1])
                    
                    if self.game=='CartPole-v0':
                        #--CartPole-v0
                        reward = self.reward(done,reward,e)
                    elif self.game=='MountainCar-v0':
                        #--MountainCar-v0
                        pos = next_state[0][0]
                        if pos>max_pos:
                            max_pos=pos
                        reward=self.reward(pos)
                        total_reward+=reward
                    elif self.game=='Acrobot-v1':
                        #--Acrobot-v1
                        reward=self.reward(done,reward,t)
                        total_reward+=reward
                    
                    state=next_state

                    if done:
                        if self.game=='CartPole-v0':
                            #--Cartpole-v0
                            print(f'Game Episode {e+1}/{n_episodes} : Score {t+1}')
                        elif self.game=='MountainCar-v0':
                            #--MountainCar-v0
                            print(f'Game Episode {e+1}/{n_episodes} : Score {int(total_reward*100)/100}, Max Pos {int(max_pos*100)/100}, Steps {t+1}')
                        elif self.game=='Acrobot-v1':
                            #--Acrobot-v1
                            print(f'Game Episode {e+1}/{n_episodes} : Score {total_reward}, Steps {t+1}')
                        else:
                            print('Write code for what to print after an episode is over')
                        break
            self.env.close()
            
        except KeyboardInterrupt:
            self.env.close()
        finally:
            self.env._max_episode_steps = self.max_steps
        
    
        


# # Training the DQN Agent (Deep Q Learner)
# agent = Agent(brain_layers=[24,24],game='CartPole-v0')
# agent.train_agent(500,render=False)
# agent.play(5,max_steps=10000)