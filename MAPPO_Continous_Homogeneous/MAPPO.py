import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import *
import numpy as np
import os

# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class Actor_RNN(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(actor_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size*N, actor_input_dim),prob.shape=(mini_batch_size*N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        prob = torch.softmax(self.fc2(self.rnn_hidden), dim=-1)
        return prob


class Critic_RNN(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(critic_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size*N, critic_input_dim), value.shape=(mini_batch_size*N, 1)
        x = self.activate_func(self.fc1(critic_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        value = self.fc2(self.rnn_hidden)
        return value


class Actor_MLP(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        # 新加
        self.fc_mean = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        # log_std is independent of state, initialized to small negative value
        self.log_std = nn.Parameter(torch.zeros(args.action_dim) - 0.5)  # or init to -0.5

        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size, episode_limit, N, actor_input_dim), prob.shape(mini_batch_size, episode_limit, N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        #下方内容代表输出连续动作        
        # mean = self.fc_mean(x)
        # log_std = self.log_std.clamp(-20, 2)
        # std = log_std.exp()
        # return mean, std
        # 下方内容代表输出离散动作
        prob = torch.softmax(self.fc3(x), dim=-1)
        return prob



class Critic_MLP(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_MLP, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size, episode_limit, N, critic_input_dim), value.shape=(mini_batch_size, episode_limit, N, 1)
        x = self.activate_func(self.fc1(critic_input))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        return value

def tanh_action_sample(dist):
    raw_action = dist.rsample()  # 用 rsample() 支持重参数
    action = torch.tanh(raw_action)
    # 修正 log_prob：加上 tanh 的导数项
    log_prob = dist.log_prob(raw_action).sum(-1)
    log_prob -= (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(-1)
    return action, log_prob

class MAPPO_MPE(nn.Module):
    def __init__(self, args,device='cpu'):
        super(MAPPO_MPE, self).__init__()
        self.device = device
        self.env = None  # 将在Runner中设置
        self.N = args.N
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode_limit = args.episode_limit
        self.rnn_hidden_dim = args.rnn_hidden_dim

        # 添加一个变量保存所有智能体ID（在Runner中设置）
        self.all_agents = []  # 将在Runner中设置，保存所有智能体ID

        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.value_coef = 1.0  # Critic loss系数（增大到1.0，确保critic能正常学习）
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_rnn = args.use_rnn
        self.add_agent_id = args.add_agent_id
        self.use_value_clip = args.use_value_clip

        # get the input dimension of actor and critic
        # 分离猎人和猎物的输入维度
        self.actor_input_dim_hunter = args.obs_dim
        self.actor_input_dim_prey = args.obs_dim
        self.critic_input_dim_hunter = args.state_dim
        self.critic_input_dim_prey = args.state_dim
        
        if self.add_agent_id:
            print("------add agent id (separate for hunters and prey)------")
            # 猎人：3个智能体，使用3维one-hot编码
            self.actor_input_dim_hunter += 3
            self.critic_input_dim_hunter += 3
            # 猎物：1个智能体，使用1维编码
            self.actor_input_dim_prey += 1
            self.critic_input_dim_prey += 1

        # 创建猎人网络（3个猎人共享）
        if self.use_rnn:
            print("------use rnn (separate networks)------")
            self.actor_hunter = Actor_RNN(args, self.actor_input_dim_hunter)
            self.critic_hunter = Critic_RNN(args, self.critic_input_dim_hunter)
            self.actor_prey = Actor_RNN(args, self.actor_input_dim_prey)
            self.critic_prey = Critic_RNN(args, self.critic_input_dim_prey)
        else:
            print("------use mlp (separate networks for hunters and prey)------")
            self.actor_hunter = Actor_MLP(args, self.actor_input_dim_hunter)
            self.critic_hunter = Critic_MLP(args, self.critic_input_dim_hunter)
            self.actor_prey = Actor_MLP(args, self.actor_input_dim_prey)
            self.critic_prey = Critic_MLP(args, self.critic_input_dim_prey)

        # 分别为猎人和猎物创建优化器
        self.ac_parameters_hunter = list(self.actor_hunter.parameters()) + list(self.critic_hunter.parameters())
        self.ac_parameters_prey = list(self.actor_prey.parameters()) + list(self.critic_prey.parameters())
        
        if self.set_adam_eps:
            print("------set adam eps (separate optimizers)------")
            self.ac_optimizer_hunter = torch.optim.Adam(self.ac_parameters_hunter, lr=self.lr, eps=1e-5)
            self.ac_optimizer_prey = torch.optim.Adam(self.ac_parameters_prey, lr=self.lr, eps=1e-5)
        else:
            self.ac_optimizer_hunter = torch.optim.Adam(self.ac_parameters_hunter, lr=self.lr)
            self.ac_optimizer_prey = torch.optim.Adam(self.ac_parameters_prey, lr=self.lr)

    def choose_action(self, obs_dict, evaluate):
        with torch.no_grad():
            # 分离猎人和猎物
            hunter_ids = [agent_id for agent_id in obs_dict.keys() if 'hunter' in agent_id]
            prey_ids = [agent_id for agent_id in obs_dict.keys() if 'prey' in agent_id]
            
            actions_dict = {}
            logprobs_dict = {}
            
            # 处理猎人（3个猎人共享网络）
            if len(hunter_ids) > 0:
                obs_hunters = []
                for agent_id in sorted(hunter_ids):  # 确保顺序一致
                    obs_hunters.append(obs_dict[agent_id])
                obs_hunters = torch.tensor(np.array(obs_hunters), dtype=torch.float32)  # shape=(n_hunters, obs_dim)
                
                hunter_inputs = [obs_hunters]
                if self.add_agent_id:
                    # 为3个猎人添加one-hot编码 [1,0,0], [0,1,0], [0,0,1]
                    hunter_inputs.append(torch.eye(3)[:len(hunter_ids)])
                
                hunter_inputs = torch.cat([x for x in hunter_inputs], dim=-1).to(self.device)
                
                # 使用猎人网络
                prob_hunters = self.actor_hunter(hunter_inputs)  # shape=(n_hunters, action_dim)
                if evaluate:
                    a_hunters = prob_hunters.argmax(dim=-1)
                    a_logprob_hunters = None
                else:
                    dist_hunters = Categorical(probs=prob_hunters)
                    a_hunters = dist_hunters.sample()
                    a_logprob_hunters = dist_hunters.log_prob(a_hunters)
                
                # 保存猎人的动作
                for i, agent_id in enumerate(sorted(hunter_ids)):
                    actions_dict[agent_id] = a_hunters[i].cpu().item()
                    if a_logprob_hunters is not None:
                        logprobs_dict[agent_id] = a_logprob_hunters[i].cpu().item()
            
            # 处理猎物（独立网络）
            if len(prey_ids) > 0:
                obs_prey = []
                for agent_id in prey_ids:
                    obs_prey.append(obs_dict[agent_id])
                obs_prey = torch.tensor(np.array(obs_prey), dtype=torch.float32)  # shape=(n_prey, obs_dim)
                
                prey_inputs = [obs_prey]
                if self.add_agent_id:
                    # 猎物只有1个，添加1维编码
                    prey_inputs.append(torch.ones((len(prey_ids), 1)))
                
                prey_inputs = torch.cat([x for x in prey_inputs], dim=-1).to(self.device)
                
                # 使用猎物网络
                prob_prey = self.actor_prey(prey_inputs)  # shape=(n_prey, action_dim)
                if evaluate:
                    a_prey = prob_prey.argmax(dim=-1)
                    a_logprob_prey = None
                else:
                    dist_prey = Categorical(probs=prob_prey)
                    a_prey = dist_prey.sample()
                    a_logprob_prey = dist_prey.log_prob(a_prey)
                
                # 保存猎物的动作
                for i, agent_id in enumerate(prey_ids):
                    actions_dict[agent_id] = a_prey[i].cpu().item()
                    if a_logprob_prey is not None:
                        logprobs_dict[agent_id] = a_logprob_prey[i].cpu().item()
            
            if len(logprobs_dict) == 0:
                return actions_dict, None
            else:
                return actions_dict, logprobs_dict

    def get_value(self, s):
        with torch.no_grad():
            s_tensor = torch.tensor(s, dtype=torch.float32)
            
            # 处理猎人的价值估计（3个猎人）
            s_hunters = s_tensor.unsqueeze(0).repeat(3, 1)  # (state_dim,)-->(3, state_dim)
            hunter_critic_inputs = [s_hunters]
            if self.add_agent_id:
                # 为3个猎人添加one-hot编码
                hunter_critic_inputs.append(torch.eye(3))
            hunter_critic_inputs = torch.cat([x for x in hunter_critic_inputs], dim=-1).to(self.device)
            v_hunters = self.critic_hunter(hunter_critic_inputs)  # shape=(3, 1)
            
            # 处理猎物的价值估计（1个猎物）
            s_prey = s_tensor.unsqueeze(0).repeat(1, 1)  # (state_dim,)-->(1, state_dim)
            prey_critic_inputs = [s_prey]
            if self.add_agent_id:
                # 猎物添加1维编码
                prey_critic_inputs.append(torch.ones((1, 1)))
            prey_critic_inputs = torch.cat([x for x in prey_critic_inputs], dim=-1).to(self.device)
            v_prey = self.critic_prey(prey_critic_inputs)  # shape=(1, 1)
            
            # 拼接所有价值：[hunter_0, hunter_1, hunter_2, prey]
            v_n = torch.cat([v_hunters, v_prey], dim=0)  # shape=(4, 1)
            return v_n.cpu().numpy().flatten()  # shape=(4,)

    def train(self, replay_buffer, total_steps):
        batch = replay_buffer.get_training_data()  # get training data
        batch = {k:v.to(self.device) for k,v in batch.items()}
        
        # 分离猎人和猎物的数据（前3个是猎人，第4个是猎物）
        # Calculate the advantage using GAE for hunters
        adv_hunter = []
        gae = 0
        with torch.no_grad():  # adv and td_target have no gradient
            deltas_hunter = batch['r_n'][:, :, :3] + self.gamma * batch['v_n'][:, 1:, :3] * (1 - batch['done_n'][:, :, :3]) - batch['v_n'][:, :-1, :3]
            for t in reversed(range(self.episode_limit)):
                gae = deltas_hunter[:, t] + self.gamma * self.lamda * gae
                adv_hunter.insert(0, gae)
            adv_hunter = torch.stack(adv_hunter, dim=1)  # shape=(batch_size, episode_limit, 3)
            v_target_hunter = adv_hunter + batch['v_n'][:, :-1, :3]
            if self.use_adv_norm:
                adv_hunter = ((adv_hunter - adv_hunter.mean()) / (adv_hunter.std() + 1e-5))
        
        # Calculate the advantage using GAE for prey
        adv_prey = []
        gae = 0
        with torch.no_grad():
            deltas_prey = batch['r_n'][:, :, 3:4] + self.gamma * batch['v_n'][:, 1:, 3:4] * (1 - batch['done_n'][:, :, 3:4]) - batch['v_n'][:, :-1, 3:4]
            for t in reversed(range(self.episode_limit)):
                gae = deltas_prey[:, t] + self.gamma * self.lamda * gae
                adv_prey.insert(0, gae)
            adv_prey = torch.stack(adv_prey, dim=1)  # shape=(batch_size, episode_limit, 1)
            v_target_prey = adv_prey + batch['v_n'][:, :-1, 3:4]
            if self.use_adv_norm:
                adv_prey = ((adv_prey - adv_prey.mean()) / (adv_prey.std() + 1e-5))

        # Get inputs for hunters and prey
        actor_inputs_hunter, critic_inputs_hunter = self.get_inputs_hunter(batch)
        actor_inputs_prey, critic_inputs_prey = self.get_inputs_prey(batch)

        # 记录总的loss值
        total_actor_loss_hunter = 0
        total_critic_loss_hunter = 0
        total_actor_loss_prey = 0
        total_critic_loss_prey = 0
        update_count = 0

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                # ============ 训练猎人网络 ============
                if self.use_rnn:
                    self.actor_hunter.rnn_hidden = None
                    self.critic_hunter.rnn_hidden = None
                    probs_now_hunter, values_now_hunter = [], []
                    for t in range(self.episode_limit):
                        prob = self.actor_hunter(actor_inputs_hunter[index, t].reshape(self.mini_batch_size * 3, -1))
                        probs_now_hunter.append(prob.reshape(self.mini_batch_size, 3, -1))
                        v = self.critic_hunter(critic_inputs_hunter[index, t].reshape(self.mini_batch_size * 3, -1))
                        values_now_hunter.append(v.reshape(self.mini_batch_size, 3))
                    probs_now_hunter = torch.stack(probs_now_hunter, dim=1)
                    values_now_hunter = torch.stack(values_now_hunter, dim=1)
                else:
                    probs_now_hunter = self.actor_hunter(actor_inputs_hunter[index])
                    values_now_hunter = self.critic_hunter(critic_inputs_hunter[index]).squeeze(-1)

                dist_now_hunter = Categorical(probs_now_hunter)
                dist_entropy_hunter = dist_now_hunter.entropy()
                a_logprob_n_now_hunter = dist_now_hunter.log_prob(batch['a_n'][index, :, :3].long())
                ratios_hunter = torch.exp(a_logprob_n_now_hunter - batch['a_logprob_n'][index, :, :3].detach())
                surr1_hunter = ratios_hunter * adv_hunter[index]
                surr2_hunter = torch.clamp(ratios_hunter, 1 - self.epsilon, 1 + self.epsilon) * adv_hunter[index]
                actor_loss_hunter = -torch.min(surr1_hunter, surr2_hunter) - self.entropy_coef * dist_entropy_hunter

                if self.use_value_clip:
                    values_old_hunter = batch["v_n"][index, :-1, :3].detach()
                    values_error_clip = torch.clamp(values_now_hunter - values_old_hunter, -self.epsilon, self.epsilon) + values_old_hunter - v_target_hunter[index]
                    values_error_original = values_now_hunter - v_target_hunter[index]
                    critic_loss_hunter = torch.max(values_error_clip ** 2, values_error_original ** 2)
                else:
                    critic_loss_hunter = (values_now_hunter - v_target_hunter[index]) ** 2

                self.ac_optimizer_hunter.zero_grad()
                ac_loss_hunter = actor_loss_hunter.mean() + self.value_coef * critic_loss_hunter.mean()
                ac_loss_hunter.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters_hunter, 10.0)
                self.ac_optimizer_hunter.step()

                # ============ 训练猎物网络 ============
                if self.use_rnn:
                    self.actor_prey.rnn_hidden = None
                    self.critic_prey.rnn_hidden = None
                    probs_now_prey, values_now_prey = [], []
                    for t in range(self.episode_limit):
                        prob = self.actor_prey(actor_inputs_prey[index, t].reshape(self.mini_batch_size * 1, -1))
                        probs_now_prey.append(prob.reshape(self.mini_batch_size, 1, -1))
                        v = self.critic_prey(critic_inputs_prey[index, t].reshape(self.mini_batch_size * 1, -1))
                        values_now_prey.append(v.reshape(self.mini_batch_size))  # shape=(mini_batch_size,)
                    probs_now_prey = torch.stack(probs_now_prey, dim=1)  # shape=(mini_batch_size, episode_limit, 1, action_dim)
                    values_now_prey = torch.stack(values_now_prey, dim=1)  # shape=(mini_batch_size, episode_limit)
                else:
                    probs_now_prey = self.actor_prey(actor_inputs_prey[index])
                    values_now_prey = self.critic_prey(critic_inputs_prey[index]).squeeze(-1).squeeze(-1)  # shape=(mini_batch_size, episode_limit)

                # squeeze掉猎物维度，使得shape从(mini_batch, episode_limit, 1, action_dim)变为(mini_batch, episode_limit, action_dim)
                probs_now_prey_squeezed = probs_now_prey.squeeze(2)  # shape=(mini_batch_size, episode_limit, action_dim)
                dist_now_prey = Categorical(probs_now_prey_squeezed)
                dist_entropy_prey = dist_now_prey.entropy()  # shape=(mini_batch_size, episode_limit)
                a_logprob_n_now_prey = dist_now_prey.log_prob(batch['a_n'][index, :, 3].long())  # 直接取第3个智能体
                ratios_prey = torch.exp(a_logprob_n_now_prey - batch['a_logprob_n'][index, :, 3].detach())
                surr1_prey = ratios_prey * adv_prey[index].squeeze(-1)
                surr2_prey = torch.clamp(ratios_prey, 1 - self.epsilon, 1 + self.epsilon) * adv_prey[index].squeeze(-1)
                actor_loss_prey = -torch.min(surr1_prey, surr2_prey) - self.entropy_coef * dist_entropy_prey

                if self.use_value_clip:
                    values_old_prey = batch["v_n"][index, :-1, 3].detach()  # 直接取第3个智能体，shape=(mini_batch_size, episode_limit)
                    values_error_clip = torch.clamp(values_now_prey - values_old_prey, -self.epsilon, self.epsilon) + values_old_prey - v_target_prey[index].squeeze(-1)
                    values_error_original = values_now_prey - v_target_prey[index].squeeze(-1)
                    critic_loss_prey = torch.max(values_error_clip ** 2, values_error_original ** 2)
                else:
                    critic_loss_prey = (values_now_prey - v_target_prey[index].squeeze(-1)) ** 2

                self.ac_optimizer_prey.zero_grad()
                ac_loss_prey = actor_loss_prey.mean() + self.value_coef * critic_loss_prey.mean()
                ac_loss_prey.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters_prey, 10.0)
                self.ac_optimizer_prey.step()

                # 累积loss值
                total_actor_loss_hunter += actor_loss_hunter.mean().item()
                total_critic_loss_hunter += critic_loss_hunter.mean().item()
                total_actor_loss_prey += actor_loss_prey.mean().item()
                total_critic_loss_prey += critic_loss_prey.mean().item()
                update_count += 1

        if self.use_lr_decay:
            self.lr_decay(total_steps)

        # 返回平均loss（猎人和猎物的平均）
        avg_actor_loss_hunter = total_actor_loss_hunter / update_count if update_count > 0 else 0
        avg_critic_loss_hunter = total_critic_loss_hunter / update_count if update_count > 0 else 0
        avg_actor_loss_prey = total_actor_loss_prey / update_count if update_count > 0 else 0
        avg_critic_loss_prey = total_critic_loss_prey / update_count if update_count > 0 else 0
        
        # 打印详细的loss信息用于调试
        if total_steps % 5000 == 0 or total_steps < 1000:
            print(f"  [Training Debug at step {total_steps}]")
            print(f"    Hunter - Actor Loss: {avg_actor_loss_hunter:.4f}, Critic Loss: {avg_critic_loss_hunter:.4f}")
            print(f"    Prey   - Actor Loss: {avg_actor_loss_prey:.4f}, Critic Loss: {avg_critic_loss_prey:.4f}")
            print(f"    Advantage Hunter range: [{adv_hunter.min().item():.4f}, {adv_hunter.max().item():.4f}]")
            print(f"    Advantage Prey range: [{adv_prey.min().item():.4f}, {adv_prey.max().item():.4f}]")
            print(f"    V_target Hunter range: [{v_target_hunter.min().item():.4f}, {v_target_hunter.max().item():.4f}]")
            print(f"    V_target Prey range: [{v_target_prey.min().item():.4f}, {v_target_prey.max().item():.4f}]")
        
        # 返回综合loss（可以根据需要调整）
        avg_actor_loss = (avg_actor_loss_hunter + avg_actor_loss_prey) / 2
        avg_critic_loss = (avg_critic_loss_hunter + avg_critic_loss_prey) / 2
        
        return avg_actor_loss, avg_critic_loss

    def lr_decay(self, total_steps):  # Trick 6: learning rate Decay
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        # 更新猎人优化器的学习率
        for p in self.ac_optimizer_hunter.param_groups:
            p['lr'] = lr_now
        # 更新猎物优化器的学习率
        for p in self.ac_optimizer_prey.param_groups:
            p['lr'] = lr_now

    def get_inputs_hunter(self, batch):
        """获取猎人的输入（前3个智能体）"""
        actor_inputs, critic_inputs = [], []
        # 只取前3个智能体（猎人）的观测
        actor_inputs.append(batch['obs_n'][:, :, :3, :])  # shape=(batch_size, episode_limit, 3, obs_dim)
        critic_inputs.append(batch['s'].unsqueeze(2).repeat(1, 1, 3, 1))  # shape=(batch_size, episode_limit, 3, state_dim)
        
        if self.add_agent_id:
            # 为3个猎人添加3x3 one-hot编码
            agent_id_one_hot = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.episode_limit, 1, 1).to(self.device)
            actor_inputs.append(agent_id_one_hot)
            critic_inputs.append(agent_id_one_hot)

        actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)  # shape=(batch_size, episode_limit, 3, actor_input_dim_hunter)
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # shape=(batch_size, episode_limit, 3, critic_input_dim_hunter)
        return actor_inputs, critic_inputs
    
    def get_inputs_prey(self, batch):
        """获取猎物的输入（第4个智能体）"""
        actor_inputs, critic_inputs = [], []
        # 只取第4个智能体（猎物）的观测
        actor_inputs.append(batch['obs_n'][:, :, 3:4, :])  # shape=(batch_size, episode_limit, 1, obs_dim)
        critic_inputs.append(batch['s'].unsqueeze(2).repeat(1, 1, 1, 1))  # shape=(batch_size, episode_limit, 1, state_dim)
        
        if self.add_agent_id:
            # 为1个猎物添加1维编码
            agent_id_one_hot = torch.ones(self.batch_size, self.episode_limit, 1, 1).to(self.device)
            actor_inputs.append(agent_id_one_hot)
            critic_inputs.append(agent_id_one_hot)

        actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)  # shape=(batch_size, episode_limit, 1, actor_input_dim_prey)
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # shape=(batch_size, episode_limit, 1, critic_input_dim_prey)
        return actor_inputs, critic_inputs

    def save_model(self, env_name, number, seed, total_steps, time_stamp):
        # 获取脚本文件所在目录的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 在脚本目录下创建models目录
        models_dir = os.path.join(current_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        # 创建以时间戳命名的子目录
        timestamp_dir = os.path.join(models_dir, time_stamp)
        os.makedirs(timestamp_dir, exist_ok=True)
        
        # 保存猎人模型（actor和critic）
        actor_hunter_path = os.path.join(timestamp_dir, f"MAPPO_actor_hunter_env_{env_name}_number_{number}_seed_{seed}_step_{int(total_steps / 1000)}k.pth")
        critic_hunter_path = os.path.join(timestamp_dir, f"MAPPO_critic_hunter_env_{env_name}_number_{number}_seed_{seed}_step_{int(total_steps / 1000)}k.pth")
        torch.save(self.actor_hunter.state_dict(), actor_hunter_path)
        torch.save(self.critic_hunter.state_dict(), critic_hunter_path)
        
        # 保存猎物模型（actor和critic）
        actor_prey_path = os.path.join(timestamp_dir, f"MAPPO_actor_prey_env_{env_name}_number_{number}_seed_{seed}_step_{int(total_steps / 1000)}k.pth")
        critic_prey_path = os.path.join(timestamp_dir, f"MAPPO_critic_prey_env_{env_name}_number_{number}_seed_{seed}_step_{int(total_steps / 1000)}k.pth")
        torch.save(self.actor_prey.state_dict(), actor_prey_path)
        torch.save(self.critic_prey.state_dict(), critic_prey_path)
        
        print(f"Models saved to {timestamp_dir}")

    def load_model(self, env_name, number, seed, step, timestamp=None):
        # 获取脚本文件所在目录的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__)) 
        # models目录路径
        models_dir = os.path.join(current_dir, "models")
        # 如果提供了时间戳，使用指定的时间戳目录
        if timestamp:
            timestamp_dir = os.path.join(models_dir, timestamp)
        else:
            # 如果没有提供时间戳但有类属性时间戳，使用类属性
            if hasattr(self, 'timestamp') and self.timestamp:
                timestamp_dir = os.path.join(models_dir, self.timestamp)
            else:
                # 如果都没有，直接使用models目录
                timestamp_dir = models_dir
        
        # 构建猎人模型路径
        actor_hunter_path = os.path.join(timestamp_dir, f"MAPPO_actor_hunter_env_{env_name}_number_{number}_seed_{seed}_step_{step}k.pth")
        critic_hunter_path = os.path.join(timestamp_dir, f"MAPPO_critic_hunter_env_{env_name}_number_{number}_seed_{seed}_step_{step}k.pth")
        
        # 构建猎物模型路径
        actor_prey_path = os.path.join(timestamp_dir, f"MAPPO_actor_prey_env_{env_name}_number_{number}_seed_{seed}_step_{step}k.pth")
        critic_prey_path = os.path.join(timestamp_dir, f"MAPPO_critic_prey_env_{env_name}_number_{number}_seed_{seed}_step_{step}k.pth")
        
        # 加载猎人模型
        self.actor_hunter.load_state_dict(torch.load(actor_hunter_path))
        self.critic_hunter.load_state_dict(torch.load(critic_hunter_path))
        print(f"加载猎人模型: {actor_hunter_path}")
        
        # 加载猎物模型
        self.actor_prey.load_state_dict(torch.load(actor_prey_path))
        self.critic_prey.load_state_dict(torch.load(critic_prey_path))
        print(f"加载猎物模型: {actor_prey_path}")