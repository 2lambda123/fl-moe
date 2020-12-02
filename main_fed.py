"""
Federated learning using a mixture of experts

based on https://github.com/edvinli/federated-learning-mixture
"""

import os.path
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from utils.sample_data import mnist_iid, mnist_iid2, mnist_noniid2, cifar_iid, cifar_iid2, cifar_noniid, cifar_noniid2
from utils.arguments import args_parser
from models.ClientUpdate import ClientUpdate
from models.Models import MLP, CNNCifar, GateCNN, GateMLP, CNNFashion, GateCNNFashion, GateCNNSoftmax, MLP2, CNNLeaf
from models.FederatedAveraging import FedAvg
from models.test_model import test_img, test_img_mix
from sys import exit
from utils.util import get_logger


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        #torch.nn.init.xavier_uniform(m.bias.data)
    elif isinstance(m,torch.nn.Linear):
        #torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

if __name__ == '__main__':

    mylogger = get_logger("fl-moe")

    args = args_parser()

    filename = args.filename
    filexist = os.path.isfile('save/' + filename)
    if(not filexist):
        with open('save/' + filename, 'a') as f1:
            f1.write('dataset;model;epochs;local_ep;num_clients;iid;p;opt;n_data;train_frac;train_gate_only;val_acc_avg_e2e;val_acc_avg_e2e_neighbour;val_acc_avg_locals;val_acc_avg_fedavg;ft_val_acc;val_acc_avg_3;val_acc_avg_rep;val_acc_avg_repft;acc_test_mix;acc_test_locals;acc_test_fedavg;ft_test_acc;ft_train_acc;train_acc_avg_locals;val_acc_gateonly;overlap;run')

            f1.write('\n')


    for run in range(args.runs):

        args.device = torch.device('cuda:{}'.format(args.gpu))

        # Create datasets
        if args.dataset == 'mnist':
            trans_mnist = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = datasets.MNIST(
                '../data/mnist/', train=True, download=True, transform=trans_mnist)
            dataset_test = datasets.MNIST(
                '../data/mnist/', train=False, download=True, transform=trans_mnist)

            if args.iid:
                dict_users = mnist_iid(dataset_train, args.num_clients)
            else:
                dict_users = mnist_noniid2(
                    dataset_train, args.num_clients, args.p)

        elif args.dataset == "femnist":

            from FemnistDataset import FemnistDataset

            # TODO: add transform
            #trans_femnist = transforms.Compose(
            #    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

            root_dir = "/proj/second-carrier-prediction/leaf/data/femnist/data/"
            dataset_train = FemnistDataset(root_dir=root_dir, train=True)
            dataset_test = FemnistDataset(root_dir=root_dir, train=False)

            # TODO: Add user sampling for the population
            dict_users = rename_keys(dataset_train.dict_users)
            dict_users_test = rename_keys(dataset_test.dict_users)


        elif args.dataset == 'cifar10':
            trans_cifar = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_train = datasets.CIFAR10(
                '../data/cifar', train=True, download=True, transform=trans_cifar)
            dataset_test = datasets.CIFAR10(
                '../data/cifar', train=False, download=True, transform=trans_cifar)

            if args.iid:
                dict_users = cifar_iid(
                    dataset_train, args.num_clients, args.n_data)
            else:
                dict_users, dict_users_test = cifar_noniid2(
                    dataset_train, dataset_test, args.num_clients, args.p, args.n_data, args.n_data_test, args.overlap)

        elif args.dataset == 'cifar100':
            trans_cifar = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_train = datasets.CIFAR100(
                '../data/cifar100', train=True, download=True, transform=trans_cifar)
            dataset_test = datasets.CIFAR100(
                '../data/cifar100', train=False, download=True, transform=trans_cifar)

            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_clients)
            else:
                dict_users, dict_users_test = cifar_noniid2(
                    dataset_train, dataset_test, args.num_clients, args.p, args.n_data, args.n_data_test, args.overlap)

        elif args.dataset == 'fashion-mnist':
            trans_fashionmnist = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            dataset_train = datasets.FashionMNIST(
                '../data/fashion-mnist', train=True, download=True, transform=trans_fashionmnist)
            dataset_test = datasets.FashionMNIST(
                '../data/fashion-mnist', train=False, download=True, transform=trans_fashionmnist)

            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_clients)
            else:
                dict_users, dict_users_test = cifar_noniid2(
                    dataset_train, dataset_test, args.num_clients, args.p, args.n_data, args.n_data_test, args.overlap)
        else:
            exit('error: dataset not available')

        img_size = dataset_train[0][0].shape

        input_length = 1
        for x in img_size:
            input_length *= x

        if (args.model == 'cnn') and (args.dataset in ['cifar10', 'cifar100']):
            net_glob_fedAvg = CNNCifar(args=args).to(args.device)
            gates_e2e_model = GateCNN(args=args).to(args.device)
            net_locals_model = CNNCifar(args=args).to(args.device)

            # opt-out fraction
            opt = np.ones(args.num_clients)
            opt_out = np.random.choice(range(args.num_clients), size=int(
                args.opt * args.num_clients), replace=False)
            opt[opt_out] = 0.0

            gates_3 = []
            gates_e2e = []
            net_locals = []
            for i in range(args.num_clients):
                gates_e2e.append(copy.deepcopy(gates_e2e_model))
                net_locals.append(copy.deepcopy(net_locals_model))

        if (args.model == 'leaf') and (args.dataset in ['cifar10', 'cifar100', 'femnist']):
            net_glob_fedAvg = CNNLeaf(args=args).to(args.device)
            gates_e2e_model = GateCNN(args=args).to(args.device)
            net_locals_model = CNNLeaf(args=args).to(args.device)

            net_glob_fedAvg.apply(weights_init)
            gates_e2e_model.apply(weights_init)
            net_locals_model.apply(weights_init)

            # opt-out fraction
            opt = np.ones(args.num_clients)
            opt_out = np.random.choice(
                range(args.num_clients),
                size=int(args.opt * args.num_clients),
                replace=False)
            opt[opt_out] = 0.0

            gates_3 = []
            gates_e2e = []
            net_locals = []

            for i in range(args.num_clients):
                gates_e2e.append(copy.deepcopy(gates_e2e_model))
                net_locals.append(copy.deepcopy(net_locals_model))

        elif (args.model == 'cnn') and (args.dataset in ['mnist', 'fashion-mnist']):
            net_glob_fedAvg = CNNFashion(args=args).to(args.device)
            gates_e2e_model = GateCNNFashion(args=args).to(args.device)
            net_locals_model = CNNFashion(args=args).to(args.device)

            # opt-out fraction
            opt = np.ones(args.num_clients)
            opt_out = np.random.choice(range(args.num_clients), size=int(
                args.opt * args.num_clients), replace=False)
            opt[opt_out] = 0.0

            gates_3 = []
            gates_e2e = []
            net_locals = []
            for i in range(args.num_clients):
                gates_e2e.append(copy.deepcopy(gates_e2e_model))
                net_locals.append(copy.deepcopy(net_locals_model))

        elif args.model == 'mlp':
            net_glob_fedAvg = MLP(
                dim_in=input_length, dim_hidden=200, dim_out=args.num_classes).to(args.device)

            #gates = []
            net_locals = []
            gates_e2e = []

            # opt-out fraction
            opt = np.ones(args.num_clients)
            opt_out = np.random.choice(range(args.num_clients), size=int(
                args.opt * args.num_clients), replace=False)
            opt[opt_out] = 0.0
            mylogger.debug(opt)
            for i in range(args.num_clients):
                gates_e2e.append(GateMLP(dim_in=input_length,
                                         dim_hidden=200, dim_out=1).to(args.device))
                net_locals.append(MLP(
                    dim_in=input_length, dim_hidden=200, dim_out=args.num_classes).to(args.device))

        else:
            exit('error: no such model')

        mylogger.debug(net_glob_fedAvg)
        for i in range(args.num_clients):
            gates_e2e[i].train()
            net_locals[i].train()

        # training
        acc_test_locals, acc_test_mix, acc_test_fedavg = [], [], []

        acc_test_finetuned_avg = []

        mylogger.info(f"Starting Federated Learning with {args.num_clients} clients for {args.epochs} rounds.")
        for iteration in range(args.epochs):
            mylogger.info(f"Round {iteration}")

            w_fedAvg = []
            alpha = []

            m = max(int(args.frac * args.num_clients), 1)

            idxs_users = np.random.choice(
                range(args.num_clients), m, replace=False)

            for idx in idxs_users:
                mylogger.debug(f"FedAvg client {idx}")

                client = ClientUpdate(args=args,
                                      train_set=dataset_train,
                                      val_set=dataset_test,
                                      idxs_train=dict_users[idx],
                                      idxs_val=dict_users_test[idx])

                if(opt[idx]):
                    # train FedAvg
                    w_glob_fedAvg, _ = client.train(net=copy.deepcopy(
                        net_glob_fedAvg).to(args.device), n_epochs=args.local_ep)

                    w_fedAvg.append(copy.deepcopy(w_glob_fedAvg))

                    # Weigh models by client dataset size
                    alpha.append(len(dict_users[idx]) / len(dataset_train))

            # update global model weights
            w_glob_fedAvg = FedAvg(w_fedAvg, alpha)

            # copy weight to net_glob
            net_glob_fedAvg.load_state_dict(w_glob_fedAvg)

        val_acc_locals, val_acc_mix = [], []
        val_acc_fedavg, val_acc_e2e = [], []
        val_acc_3, val_acc_rep, val_acc_repft, val_acc_ft = [], [], [], []
        val_acc_e2e_neighbour, val_acc_gateonly = [], []
        train_acc_ft, train_acc_locals = [], []
        acc_test_l, acc_test_m = [], []
        gate_values = []
        finetuned = []

        mylogger.info("Starting finetuning")
        for idx in range(args.num_clients):

            client = ClientUpdate(args=args,
                                  train_set=dataset_train,
                                  val_set=dataset_test,
                                  idxs_train=dict_users[idx],
                                  idxs_val=dict_users_test[idx])

            # finetune FedAvg for every client
            mylogger.debug(f"Finetuning for client {idx}")

            # TODO: Remove magical constants
            wt, _, val_acc_finetuned, train_acc_finetuned = client.train_finetune(
                net=copy.deepcopy(net_glob_fedAvg).to(args.device),
                n_epochs=200,
                learning_rate=args.ft_lr)

            val_acc_ft.append(val_acc_finetuned)
            train_acc_ft.append(train_acc_finetuned)

            ft_net = copy.deepcopy(net_glob_fedAvg)
            ft_net.load_state_dict(wt)
            finetuned.append(ft_net)

            # train local model
            # TODO: Remove magical constants
            mylogger.debug(f"Training local model for client {idx}")
            w_l, _, val_acc_l, train_acc_l = client.train_finetune(
                net=net_locals[idx].to(args.device),
                n_epochs=200,
                learning_rate=args.local_lr)

            net_locals[idx].load_state_dict(w_l)
            val_acc_locals.append(val_acc_l)
            train_acc_locals.append(train_acc_l)

        mylogger.info("Starting MoE trainings")
        for idx in range(args.num_clients):

            client = ClientUpdate(args=args,
                                  train_set=dataset_train,
                                  val_set=dataset_test,
                                  idxs_train=dict_users[idx],
                                  idxs_val=dict_users_test[idx])

            mylogger.debug(f"Training mixtures for client {idx}")

            _, _, val_acc_e2e_k, _ = client.train_mix(
                net_local=copy.deepcopy(net_locals[idx]).to(args.device),
                net_global=copy.deepcopy(net_glob_fedAvg).to(args.device),
                gate=copy.deepcopy(gates_e2e[idx]),
                train_gate_only=args.train_gate_only,
                n_epochs=200,
                early_stop=True,
                learning_rate=args.moe_lr)

            val_acc_e2e.append(val_acc_e2e_k)

            # evaluate FedAvg on local dataset
            val_acc_fed, _ = client.validate(
                net=net_glob_fedAvg.to(args.device))
            val_acc_fedavg.append(val_acc_fed)

        # Calculate validation and test accuracies

        val_acc_avg_locals = sum(val_acc_locals) / len(val_acc_locals)

        train_acc_avg_locals = sum(train_acc_locals) / len(train_acc_locals)

        val_acc_avg_e2e = sum(val_acc_e2e) / len(val_acc_e2e)
        #val_acc_avg_e2e = np.nan

        #val_acc_avg_e2e_neighbour = sum(val_acc_e2e_neighbour) / len(val_acc_e2e_neighbour)
        val_acc_avg_e2e_neighbour = np.nan

        #val_acc_avg_3 = sum(val_acc_3) / len(val_acc_3)
        val_acc_avg_3 = np.nan

        #val_acc_avg_gateonly = sum(val_acc_gateonly) / len(val_acc_gateonly)
        val_acc_avg_gateonly = np.nan

        #val_acc_avg_rep = sum(val_acc_rep) / len(val_acc_rep)
        val_acc_avg_rep = np.nan

        #val_acc_avg_repft = sum(val_acc_repft) / len(val_acc_repft)
        val_acc_avg_repft = np.nan

        val_acc_avg_fedavg = sum(val_acc_fedavg) / len(val_acc_fedavg)

        ft_val_acc = sum(val_acc_ft) / len(val_acc_ft)
        ft_train_acc = sum(train_acc_ft) / len(train_acc_ft)
        ft_test_acc = np.nan

        with open('save/' + filename, 'a') as f1:
            f1.write('{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}'.format(
                args.dataset, args.model, args.epochs, args.local_ep, args.num_clients, args.iid, args.p, args.opt, args.n_data, args.train_frac,
                args.train_gate_only, val_acc_avg_e2e, val_acc_avg_e2e_neighbour, val_acc_avg_locals, val_acc_avg_fedavg, ft_val_acc, val_acc_avg_3,
                val_acc_avg_rep, val_acc_avg_repft, acc_test_mix, acc_test_locals, acc_test_fedavg, ft_test_acc, ft_train_acc, train_acc_avg_locals,
                val_acc_avg_gateonly, args.overlap, run))
            f1.write("\n")
        mylogger.info("Done")
