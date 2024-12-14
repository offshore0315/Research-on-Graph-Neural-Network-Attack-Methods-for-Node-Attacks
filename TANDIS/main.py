# bottom-up I understand. Let's go step-by-step
import random
import numpy as np
import torch
import os

from cmd_args import cmd_args

from env import GraphEnv
from agent import Agent
import pickle

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import utils
from attack_models.grl_models import AttackModel

def get_attack_model (saved_model):#, embs=False):
    with open('%s-args.pkl' % saved_model, 'rb') as f:
        base_args = pickle.load(f)

    base_args['device'] = cmd_args.device

    return AttackModel(base_args).to(cmd_args.device) #if (not (embs)) else AttackEmbModel(base_args)

def load_base_model():
    assert cmd_args.saved_model is not None

    base_model = get_attack_model (cmd_args.saved_model)
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    base_model.load_state_dict(torch.load(cmd_args.saved_model+ '.model', map_location=map_location))
    base_model.eval()
    return base_model

def load_emb_model():
    assert cmd_args.saved_emb_model is not None

    # This could as well be a transferable different model entirely 
    emb_model = get_attack_model (cmd_args.saved_emb_model)#, embs=True)
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    emb_model.load_state_dict(torch.load(cmd_args.saved_emb_model+ '.model', map_location=map_location))
    emb_model.eval()
    return emb_model

def init_setup():
    # taking the first graph since node-level attack only right now.
    root = os.getcwd()
    if (cmd_args.dataset in ["cora", "citeseer", "pubmed"]):
        data_root = root + "/data/Planetoid/"
        dataset = Planetoid(root=data_root, name=cmd_args.dataset.capitalize(), transform=NormalizeFeatures())[0].to(cmd_args.device)
    else:
        data_root = root + "/data/"
        dataset = utils.load_other_datasets(data_root)
    idx_file_end = cmd_args.down_task + "_" + str(cmd_args.seed) + ".pt"
    if (cmd_args.lcc):
        dataset = utils.get_largest_component(dataset)
        idx_file_end  = "lcc_" + idx_file_end
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    dataset = utils.preprocess(dataset, data_root, cmd_args.down_task)
    idx_train = dataset.train_mask.T
    idx_val = dataset.val_mask.T
    idx_test = dataset.test_mask.T
    try:
        idx_train2 = torch.load(data_root + cmd_args.dataset.capitalize() + "/train_" + idx_file_end)
        idx_val2 = torch.load(data_root + cmd_args.dataset.capitalize() + "/val_" + idx_file_end)
        idx_test2 = torch.load(data_root + cmd_args.dataset.capitalize() + "/test_" + idx_file_end)
        print(torch.all(idx_train == idx_train2))
        print(torch.all(idx_val == idx_val2))
        print(torch.all(idx_test == idx_test2))
    except:
        torch.save(idx_train, data_root + cmd_args.dataset.capitalize() + "/train_" + idx_file_end)
        torch.save(idx_val, data_root + cmd_args.dataset.capitalize() + "/val_" + idx_file_end)
        torch.save(idx_test, data_root + cmd_args.dataset.capitalize() + "/test_" + idx_file_end)
    # exit()
    return dataset, idx_train, idx_val, idx_test

if __name__ == '__main__':
    print(cmd_args)
    data, idx_train, idx_val, idx_test = init_setup()
    print (idx_train.shape, idx_val.shape, idx_test.shape)
    # exit()
    # just prune the dict_of_lists to a targetted setting 
    # and even for candidate sampling
    candidate_nodes = None

    if cmd_args.phase == 'train':
        idx_train = torch.cat ((idx_train, idx_val))
        emb_model = load_emb_model()
        env = GraphEnv(cmd_args.down_task, data.x, data.y, data.edge_index, emb_model)
        agent = Agent(env, data.x, idx_train, idx_val, candidate_nodes)    
        print("============ Training ================")
        agent.train()
    else:
        if (cmd_args.target_perc is not None):
            data, idx_test = utils.sample_small_test (data, idx_test, frac=cmd_args.target_perc) 
            root = os.getcwd()
            data_root = root + "/data/Planetoid/"
            if (cmd_args.lcc):
                idx_file_end = "smpld_lcc_" + cmd_args.down_task + "_" + str(cmd_args.seed) + ".pt"
            else:
                idx_file_end = "smpld_" + cmd_args.down_task + "_" + str(cmd_args.seed) + ".pt"
            try:
                idx_test2 = torch.load(data_root + cmd_args.dataset.capitalize() + "/test_" + idx_file_end)
                print(torch.all(idx_test == idx_test2))
            except:
                torch.save(idx_test, data_root + cmd_args.dataset.capitalize() + "/test_" + idx_file_end)
            
        if (cmd_args.save_sols_only):
            env = GraphEnv(cmd_args.down_task, data.x, data.y, data.edge_index, None, base_model=None)
            agent = Agent(env, data.x, idx_train, idx_val, candidate_nodes)
            agent.net.load_state_dict(torch.load(cmd_args.save_dir + '/epoch-best.model'))
            agent.eval(idx_test)
        else:
            data = data.to(cmd_args.device)
            base_model = load_base_model()
            emb_model = load_emb_model()
            # emb_model None??
            env = GraphEnv(cmd_args.down_task, data.x, data.y, data.edge_index, emb_model, base_model=base_model)
            
            agent = Agent(env, data.x, idx_train, idx_val, candidate_nodes)
            z = base_model (data.x, data.edge_index)
            print (z.shape)
            # idx_test = torch.load(f'attack_models/node_classification/cora/CAS_test_idx.pt')
            # data.test_y = torch.load(f'attack_models/node_classification/cora/CAS_test_y.pt')
            pred_y = base_model.predict(z[idx_test.T], classify=True)
            print (pred_y.shape, data.test_y.shape)
            acc_test = (pred_y == data.test_y)
            test_ids = idx_test[acc_test > 0]
            num_wrong = torch.sum (acc_test == 0)
            print(len(test_ids) / float(len(idx_test)))
            # 
            # evaluated on meta_list and not attack_list which is strange 
            agent.net.load_state_dict(torch.load(cmd_args.save_dir + '/epoch-best.model'))
            print("======evaling=====")
            agent.eval(test_ids, num_wrong=num_wrong)

