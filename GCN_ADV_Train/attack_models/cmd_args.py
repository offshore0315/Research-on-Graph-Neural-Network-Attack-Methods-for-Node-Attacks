import argparse

cmd_opt = argparse.ArgumentParser(description='Argparser for training attack model')
cmd_opt.add_argument("-dataset", type=str, default=None)
cmd_opt.add_argument('-layer', type=str, default='gcn')
cmd_opt.add_argument('-model_name', type=str, default='Deep')
cmd_opt.add_argument('-model_save_name', type=str, default=None)
cmd_opt.add_argument('-down_task', type=str, default="node_classification")

cmd_opt.add_argument('-device', type=str, default="cpu")

cmd_opt.add_argument('-dropout_rate', type=float, default=0.5)
cmd_opt.add_argument('-hidden_layers', type=int, nargs='+', default=[32, 32])


cmd_opt.add_argument('-batch_size', type=int, default=10)
cmd_opt.add_argument('-n_epochs', type=int, default=200)
cmd_opt.add_argument('-lr', type=float, default=0.01)
cmd_opt.add_argument('-patience', type=float, default=10)
cmd_opt.add_argument('-save_embs', type=int, default=1)
cmd_opt.add_argument('-seed', type=int, default=123)

cmd_opt.add_argument('-directed', type=int, default=1)

cmd_opt.add_argument('-lcc', action='store_true')

cmd_args, _ = cmd_opt.parse_known_args()
