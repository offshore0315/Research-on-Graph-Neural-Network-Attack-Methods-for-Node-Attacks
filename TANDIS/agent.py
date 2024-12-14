import random
import numpy as np

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from cmd_args import cmd_args

from tqdm import tqdm
from copy import deepcopy
from nstep_replay_mem import NstepReplayMem, NstepReplayMemCell
from my_dqn import TStepDQN
from my_dqn import DQN

from collections import deque

class Agent(object):
    def __init__(self, env, features, idx_train, idx_val, candidate_nodes):
        self.features = features
        self.idx_train = idx_train
        self.idx_val = idx_val
        # if candidate_nodes is None then consider every node otherwise consider only from this 
        self.candidate_nodes = candidate_nodes
        # parameter sharing
        self.mem_pool = NstepReplayMemCell(memory_size=500000)#, n_steps=cmd_args.budget)
        self.env = env 

        # parameter sharing
        self.net = DQN(self.env.orig_graph.cpu(), cmd_args.dqn_hidden, features.cpu(), candidate_nodes, mu_type=cmd_args.mu_type, embed_dim=cmd_args.embed_dim)
        self.old_net = DQN(self.env.orig_graph.cpu(), cmd_args.dqn_hidden, features.cpu(), candidate_nodes, mu_type=cmd_args.mu_type, embed_dim=cmd_args.embed_dim)

        # if cmd_args.device == 'cuda':
        #     self.net = self.net.cuda()
        #     self.old_net = self.old_net.cuda()

        self.eps_start = 0.99
        self.eps_end = 0.05
        self.eps_step = 1000
        self.burn_in = 10     
        self.step = 0        
        self.pos = 0
        self.best_eval = None
        self.take_snapshot()

    def uniformRandActions(self, target_nodes):
        act_list = []
        offset = 0
        for i in range(len(target_nodes)):
            v = target_nodes[i]
            while (v == target_nodes[i]):
                v = np.random.randint(self.features.shape[0])
                if (self.env.directed):
                    d = np.random.choice([-1, 1])
                else:
                    d = 1
            act_list.append(v*d)
        return act_list

    def take_snapshot(self):
        self.old_net.load_state_dict(self.net.state_dict())

    def make_actions(self, greedy=False):
        t = self.step
        self.eps = max (self.eps_end, self.eps_start**t)
        # self.eps = self.eps_end + max(0., (self.eps_start - self.eps_end)
        #         * (self.eps_step - t) / self.eps_step)

        if random.random() < self.eps and not greedy:
            actions = self.uniformRandActions(self.env.target_nodes)
        else:
            cur_state = self.env.getStateRef()
            # parameter sharing
            actions, values = self.net(cur_state, None, greedy_acts=True)
            actions = list(actions.cpu().numpy())

        return actions

    def batch_sample (self):
        if (self.pos + 1) * cmd_args.batch_size > len(self.idx_train):
            self.pos = 0
            random.shuffle(self.idx_train)

        selected_idx = self.idx_train[self.pos * cmd_args.batch_size : (self.pos + 1) * cmd_args.batch_size]
        self.pos += 1
        self.env.setup(selected_idx)

    def run_simulation(self):
        
        # while not self.env.isTerminal():
        # no need to go till terminality 
        # rather fix a sample size and run only till that
        for j in range(cmd_args.sample_size):
            list_at = self.make_actions()
            list_st = self.env.cloneState()

            self.env.step(list_at)
            if (cmd_args.reward_state == "marginal"):
                # Marginal reward (at every step)
                rewards = self.env.rewards
            elif (cmd_args.reward_state == "final"):
                # Final one-shot reward
                rewards = self.env.rewards
            # budget-agnostic training:
            s_prime = self.env.cloneState()
            # s_prime = self.env.cloneState() if (not(self.env.isTerminal())) else None
            #print (list_st, list_at, rewards, s_prime, [self.env.isTerminal()] * len(list_at), t)
            # parameter sharing (budget-agnostic training)
            self.mem_pool.add_list(list_st, list_at, rewards, s_prime) #, [self.env.isTerminal()] * len(list_at))#, t)            
        
    def train(self):
        pbar = tqdm(range(self.burn_in), unit='batch')
        states_queue = deque([], cmd_args.q_nstep)
        actions_queue = deque([], cmd_args.q_nstep)
        rewards_queue = deque([], cmd_args.q_nstep)
        for p in pbar:
            self.batch_sample()
            # epsilon-greedy actions by default
            list_at = self.make_actions()
            list_st = self.env.cloneState()

            self.env.step(list_at)
            rewards = self.env.rewards

            states_queue.append(list_st)
            actions_queue.append(list_at)
            rewards_queue.append(rewards)

            if (p >= cmd_args.q_nstep):
                s_tp1 = self.env.cloneState()
                s_tmn = states_queue[0]
                a_tmn = actions_queue[0]
                rewards_sum = np.stack(list(rewards_queue)).sum(axis=0)
                # assuming always marginal (cmd_args.reward_state not considered) <= budget-agnostic training
                self.mem_pool.add_list(s_tmn, a_tmn, rewards_sum, s_tp1) 

        ebar = tqdm(range(1, cmd_args.num_epds+1), unit='epds')
        # # To set samples
        tbar = tqdm(range(1, cmd_args.sample_size+1), unit='steps')
        # ebar = range(1, cmd_args.num_epds+1)
        # To set samples
        # tbar = range(1, cmd_args.sample_size+1)
        optimizer = optim.Adam(self.net.parameters(), lr=cmd_args.learning_rate)
    
        print("ebar len = " + str(len(ebar)))
        for self.episode in ebar:
            print (self.episode)
            print ("::::::::::::")
            states_queue = deque([], cmd_args.q_nstep)
            actions_queue = deque([], cmd_args.q_nstep)
            rewards_queue = deque([], cmd_args.q_nstep)
            # if self.episode % 10 == 0:
            #     self.eval(self.idx_val)
            for self.step in tbar:
                print (self.step)
                self.batch_sample()

                # epsilon-greedy actions by default
                list_at = self.make_actions()
                list_st = self.env.cloneState()

                self.env.step(list_at)
                rewards = self.env.rewards

                states_queue.append(list_st)
                actions_queue.append(list_at)
                rewards_queue.append(rewards)

                if (self.step > cmd_args.q_nstep):
                    s_tp1 = self.env.cloneState()
                    s_tmn = states_queue[0]
                    a_tmn = actions_queue[0]
                    rewards_sum = np.stack(list(rewards_queue)).sum(axis=0)
                    # assuming always marginal (cmd_args.reward_state not considered) <= budget-agnostic training
                    self.mem_pool.add_list(s_tmn, a_tmn, rewards_sum, s_tp1) 

                    if self.step % 3 == 0:
                        self.take_snapshot()
                    
                    list_st, list_at, list_rt, list_s_primes = self.mem_pool.sample(batch_size=cmd_args.batch_size)
                    list_target = torch.Tensor(list_rt)
                    if cmd_args.device == 'cuda':
                        list_target = list_target.cuda()

                    # parameter sharing
                    # _, q_t_plus_n = self.old_net(list_s_primes, None)
                    # _, q_rhs = self.old_net.node_greedy_actions(list_s_primes, q_t_plus_n)
                    _, q_rhs = self.old_net(list_s_primes, None, greedy_acts=True)
                    list_target += cmd_args.discount * q_rhs

                    list_target = Variable(list_target.view(-1, 1))
                    
                    # parameter sharing
                    _, q_sa = self.net(list_st, list_at)
                    q_sa = torch.cat(q_sa, dim=0)
                    loss = F.mse_loss(q_sa, list_target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    #print (loss)
                    print ('eps: %.5f, loss: %0.5f, q_val: %.5f' % (self.eps, loss, torch.mean(q_sa).item()))
                    pbar.set_description('eps: %.5f, loss: %0.5f, q_val: %.5f' % (self.eps, loss, torch.mean(q_sa).item()) )
        # self.eval(self.idx_val)
        torch.save(self.net.state_dict(), cmd_args.save_dir + '/epoch-best.model')

    def eval(self, test_ids, num_wrong=0):
        #self.env.printState ()
                    
        self.env.setup(test_ids)
        if (cmd_args.save_sols_only):
            import time
            sols_df = {"tin": list(range(self.env.targets.shape[0])), "t": self.env.target_nodes.tolist()}
            for b in range(cmd_args.budget):
                sols_df["u_" + str(b+1)] = []
            start_time = time.time()

        while not self.env.isTerminal():
            print("running")
            list_at = self.make_actions(greedy=True)
            self.env.step(list_at)
            if (cmd_args.save_sols_only):
                print ("------------", self.env.n_step, time.time() - start_time)
                for j, t  in enumerate(self.env.target_nodes):
                    v, direction = np.abs(list_at[j]), np.sign(list_at[j])
                    sols_df["u_" + str(self.env.n_step)].append(v)
            else:
                acc = self.env.perb_accuracies
                acc = np.sum(acc) / (len(test_ids) + num_wrong)
                #print("Perturbation:")
                #for i in range(len(test_ids)):
                #    print ([e for e in self.env.perb_list[i].get_node_pairs()])
                #print()
                print("here")
                print('%d | Average test: targets %d, acc %.5f' % (self.env.n_step, len(self.env.target_nodes), acc))
                self.env.update_targets()

        if (cmd_args.save_sols_only):
            import pandas as pd
            save_file = f'test_results/{cmd_args.down_task}/{cmd_args.dataset}/{cmd_args.save_sols_file}_{cmd_args.budget}.csv'
            pd.DataFrame(sols_df).to_csv(save_file, index=False)
            return
        #print ("PERTURBATION:::")
        #self.env.printState ()
        acc = self.env.perb_accuracies
        acc = np.sum(acc) / (len(test_ids) + num_wrong)
        print('%d | Average test: acc %.5f' % (self.env.n_step, acc))

        if (cmd_args.phase == 'train') and ((self.best_eval is None) or (acc < self.best_eval)):
            print('----saving to best attacker since this is the best attack rate so far.----')
            torch.save(self.net.state_dict(), cmd_args.save_dir + '/epoch-best.model')
            with open(cmd_args.save_dir + '/epoch-best.txt', 'w') as f:
                f.write('%.4f\n' % acc)
            with open(cmd_args.save_dir + '/attack_solution.txt', 'w') as f:
                for i in range(len(test_ids)):
                    f.write(str(torch.tensor(test_ids[i])) + ": [")
                    for e in self.env.perb_list[i].get_node_pairs():
                        f.write('(%d %d)' % (e[0], e[1]))
                    f.write('] succ: %d\n' % (self.env.perb_accuracies[i]))
            self.best_eval = acc
