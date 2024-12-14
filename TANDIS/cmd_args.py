import argparse
import pickle 
import numpy as np

cmd_opt = argparse.ArgumentParser(description='Argparser for graph attack via neighborhood distortion')
cmd_opt.add_argument('-data_folder', type=str, default=None, help='data folder')
cmd_opt.add_argument('-saved_model', type=str, default=None, help='saved model')
cmd_opt.add_argument('-model_name', type=str, default=None, help='Architecture model name if not default')
cmd_opt.add_argument('-save_dir', type=str, default=None, help='save folder')
cmd_opt.add_argument('-device', type=str, default='cpu', help='cpu/cuda')
cmd_opt.add_argument('-phase', type=str, default='test', help='train/test')
cmd_opt.add_argument('-down_task', type=str, default='node_classification', help='prediction task')
cmd_opt.add_argument('-logfile', type=str, default=None, help='log')
cmd_opt.add_argument('-saved_emb_model', type=str, default=None, help='saved embedding model')
cmd_opt.add_argument('-nprocs', type=int, default=20, help='n_procs')
cmd_opt.add_argument('-target_perc', type=float, default=None, help='sample perc set of targets')
cmd_opt.add_argument('-lcc', action='store_true')

cmd_opt.add_argument('-batch_size', type=int, default=50, help='minibatch size')
cmd_opt.add_argument('-seed', type=int, default=np.random.randint(100), help='seed')
cmd_opt.add_argument('-gm', default='mean_field', help='mean_field/loopy_bp/gcn')
cmd_opt.add_argument('-max_lv', type=int, default=2, help='max rounds of message passing')

cmd_opt.add_argument('-learning_rate', type=float, default=0.001, help='init learning_rate')
cmd_opt.add_argument('-weight_decay', type=float, default=5e-4, help='weight_decay')
cmd_opt.add_argument('-dropout', type=float, default=0.5, help='dropout rate')

# for node classification
cmd_opt.add_argument('-dataset', type=str, default=None, help='citeseer/cora/pubmed')
cmd_opt.add_argument('-feature_dim', type=int, default=None, help='node feature dim')
cmd_opt.add_argument('-num_class', type=int, default=None, help='# classes')

# for attack 
cmd_opt.add_argument('-num_epds', type=int, default=100000, help='rl training steps')
cmd_opt.add_argument('-sample_size', type=int, default=10, help='sample size')
cmd_opt.add_argument('-discount', type=float, default=0.9, help='Discount factor (gamma)')
cmd_opt.add_argument('-q_nstep', type=int, default=2, help='n-step q-learning')
cmd_opt.add_argument('-reward_state', type=str, default="final", help='final reward or marginal')
cmd_opt.add_argument('-reward_type', type=str, default=None, help='emb_silh')
cmd_opt.add_argument('-base_model_dump', type=str, default=None, help='saved base model')
cmd_opt.add_argument('-budget', type=int, default=1, help='number of modifications allowed')
cmd_opt.add_argument('-mu_type', type=str, default="preT_embeds", help='e2e_embeds/preT_embeds/hand-crafted')
cmd_opt.add_argument('-num_hops', type=int, default=2, help='Number of hops in state representation')
cmd_opt.add_argument('-directed', type=int, default=0, help='directed or not')
cmd_opt.add_argument('-embeds', type=str, default=None, help='location for pre-trained embeddings')
cmd_opt.add_argument('-dqn_hidden', type=int, default=16, help='DQN hidden layer size')
cmd_opt.add_argument('-embed_dim', type=int, default=16, help='dimension of end-to-end GCN ')
cmd_opt.add_argument('-nbrhood', type=str, default="20-NN", help='neighborhood function')

cmd_opt.add_argument('-save_sols_only', action='store_true')
cmd_opt.add_argument('-save_sols_file', type=str, default='sols')

cmd_args, _ = cmd_opt.parse_known_args()

# print(cmd_args)
