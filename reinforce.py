import torch, numpy as np

class Reinforce:
    def __init__(self,env, model, optimizer, gamma, episodes, device):
        self.env = env
        self.model = model.to(device)
        self.optimizer = optimizer
        self.gamma = gamma
        self.episodes = episodes
        self.device = device

    def select_action(self, state):
        probs = self.model(state)
        distribution = torch.distributions.Categorical(probs)
        action = distribution.sample().item()
        return action

    def update_model(self, episode_data):
        states = torch.stack(episode_data["states"])
        actions = torch.LongTensor(episode_data["actions"]).to(self.device)
        rewards = torch.FloatTensor(episode_data["rewards"]).to(self.device)

        #TODO: Check if there is a need to shuffle the data to diminish the correlation between samples

        discounted_rewards = []
        for i in range(len(rewards)):
            current_reward = 0.0
            for j, k in enumerate(rewards[i:]):
                current_reward += (self.gamma ** j) * k
            discounted_rewards.append(current_reward)

        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)

        for state, action, g in zip(states, actions, discounted_rewards):
            probs = self.model(state)
            distribution = torch.distributions.Categorical(probs)
            log_prob = distribution.log_prob(action)
            loss = -log_prob * g
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()



    def train(self):
        acum_reward = []
        for episode in range(self.episodes):
            state, _ = self.env.reset()
            states, actions, rewards = [], [], []
            done = False
            truncated = False
            while not done and not truncated:
                state = torch.FloatTensor(state).to(self.device)
                action = self.select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state
            epsode_data = {
                "states": states,
                "actions": actions,
                "rewards": rewards}
            self.update_model(epsode_data)
            acum_reward.append(sum(rewards))
            print(f"Episode {episode} finished with reward {acum_reward[-1]}")
        return acum_reward
   