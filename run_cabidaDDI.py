import argparse

from chemicalx.data import DrugCombDB, DrugComb, DrugbankDDI, TwoSides
from class_resolver import Resolver 
from chemicalx.models.base import Model
import torch
import torch.nn.functional as F
import os
import copy
from torch import nn
import pandas as pd
from torchdrug.layers import MaxReadout, GraphIsomorphismConv

from chemicalx_.chemicalx.compat import PackedGraph
from chemicalx.constants import TORCHDRUG_NODE_FEATURES
from chemicalx.data import DrugPairBatch, BatchGenerator
from chemicalx.models import Model
from chemicalx.pipeline import metric_resolver


desired_num_threads = 5
torch.set_num_threads(desired_num_threads)

__all__ = ["CabidaDDI"]


class ContextModule(torch.nn.Module):
    def __init__(self, context_dim, emb_dim, latent_dim=64):
        super(ContextModule, self).__init__()
        self.fc1 = torch.nn.Linear(context_dim, latent_dim)
        self.fc2 = torch.nn.Linear(latent_dim, emb_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CabidaDDI(Model):
    def __init__(
        self,
        *,
        molecule_channels: int = TORCHDRUG_NODE_FEATURES,
        num_gcn_layers: int=5,
        dropout_rate: float = 0.2,
        context_dim: int = 84,
        molecule_emb_dim: int = 300,
        output_dim: int = 1,
    ):
        super(CabidaDDI, self).__init__()
        # self.gcn_layer_hidden_size = gcn_layer_hidden_size
        self.num_gcn_layers = num_gcn_layers
        self.molecule_emb_dim = molecule_emb_dim
        self.output_dim = output_dim
        self.readout = MaxReadout()

        ''' Molecule Encoder '''
        self.gin_0 = GraphIsomorphismConv(molecule_channels, self.molecule_emb_dim, batch_norm=True)
        self.gin = torch.nn.ModuleList(
                GraphIsomorphismConv(self.molecule_emb_dim, self.molecule_emb_dim, batch_norm=True)
                for _ in range(num_gcn_layers-1)
            )
        self.batch_norm = torch.nn.ModuleList(
                torch.nn.BatchNorm1d(self.molecule_emb_dim)
                for _ in range(num_gcn_layers)
            )

        ''' Context Encoder '''
        self.context_encoder = ContextModule(context_dim, molecule_emb_dim)

        ''' Combining Module '''
        self.comb_drug_layer = torch.nn.Sequential(
            nn.Linear(molecule_emb_dim*2, 300),
        )

        ''' If Attention mode, add attention layer '''
        # Drug -> Context + Context -> Drug
        self.attn_c2d_layer = nn.MultiheadAttention(embed_dim=molecule_emb_dim, num_heads=1, dropout=dropout_rate)
        self.attn_d2c_layer = nn.MultiheadAttention(embed_dim=molecule_emb_dim, num_heads=1, dropout=dropout_rate)
        self.final = torch.nn.Sequential(
            nn.Linear(molecule_emb_dim*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid(),
        )

            
    def unpack(self, batch: DrugPairBatch):
        """Return the left drug molecules, and right drug molecules."""
        return (
            batch.context_features,
            batch.drug_molecules_left,
            batch.drug_molecules_right,
        )

    def _forward_molecules(self, molecules: PackedGraph) -> torch.FloatTensor:

        for layer in range(self.num_gcn_layers):
            if layer == 0:
                h = self.gin_0(molecules, molecules.data_dict["atom_feature"].float())
                h = self.batch_norm[layer](h)
            else:
                h = self.gin[layer-1](molecules, h)
                h = self.batch_norm[layer](h)
                if layer != self.num_gcn_layers-1:
                    h = F.relu(h)
        h = self.readout(molecules, h)
        return h


    def forward(self, context_features:torch.FloatTensor, molecules_left: PackedGraph, molecules_right: PackedGraph) -> torch.FloatTensor:
        features_left = self._forward_molecules(molecules_left)
        features_right = self._forward_molecules(molecules_right)
        
        context_features = self.context_encoder(context_features)
        drug_features = torch.cat((features_left, features_right), dim=1)
        drug_features = self.comb_drug_layer(drug_features)
        ''' Drug -> Context + Context -> Drug '''
        c2d, _ = self.attn_c2d_layer(drug_features, context_features, context_features)
        d2c, _ = self.attn_d2c_layer(context_features, drug_features, drug_features)
        x = torch.cat((c2d, d2c), dim=1)
        x = torch.cat((drug_features,context_features,x), dim=1)
        x = self.final(x)
        
        return x


def rertun_batch_generator(dataset, args, subdataset):
    generator = BatchGenerator(
        batch_size=args.batch_size,
        context_features=args.context_features,
        drug_features=True,
        drug_molecules=True,
        context_feature_set=dataset.get_context_features(),
        drug_feature_set=dataset.get_drug_features(),
        labeled_triples=subdataset,
        label="label")
    return generator


def return_dataloader(dataset, args, seed=0):
    triples = dataset.get_labeled_triples()

    train, test = triples.train_test_split(train_size=0.8, random_state=seed)
    train, valid = train.train_test_split(train_size=0.875, random_state=seed)

    train_loader = rertun_batch_generator(dataset, args, train)
    valid_loader = rertun_batch_generator(dataset, args, valid)
    test_loader = rertun_batch_generator(dataset, args, test)

    return train_loader, valid_loader, test_loader


def parse_arguments():
    parser = argparse.ArgumentParser(description='Runing CabidaDDI')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for processing')
    parser.add_argument('--dataset', type=str, default='drugcombdb', help='Dataset argument')
    parser.add_argument('--metrics', nargs='+', default=['auroc', 'auprc', 'f1', 'acc'], help='Metrics argument')
    parser.add_argument('--lr', type=float, default=0.001, help='Optimizer kwargs argument')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs argument')
    parser.add_argument('--device', type=int, default=0, help='Device to use for processing')
    parser.add_argument('--repeat', type=int, default=10, help='Seed argument')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_arguments()

    dataset_dict = {'drugcombdb': DrugCombDB(),'drugcomb': DrugComb(),'drugbankddi': DrugbankDDI(),'twosides': TwoSides()}
    dataset = dataset_dict[args.dataset]
  
    model_resolver = Resolver.from_subclasses(base=Model)

    device = args.device
    batch_size = args.batch_size
    dataname = args.dataset
    model_name = args.model
    metrics = args.metrics
    lr = args.lr
    epochs = args.epochs

    metric_dict = {name: metric_resolver.lookup(name) for name in metrics}    
    seeds = range(args.repeat)
    
    os.makedirs(f'baseline_results/{dataname}', exist_ok=True)

    for i in seeds:
        torch.manual_seed(0)

        performance = pd.DataFrame(columns=['rmse','mse','pearson','spearman'])
        loss_fn = nn.MSELoss()
        task_type = 'regression'
        label_type = args.label_name

        model = CabidaDDI(context_dim=dataset.num_contexts)
        model = model.to(device)

        train_loader, valid_loader, test_loader = return_dataloader(dataset, args, seed=i)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()

        best_auc = 0
        for e in range(epochs):
            model.train()
            train_loss = 0
            valid_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                batch = batch.to(device)
                pred = model(*model.unpack(batch))
                loss = loss_fn(pred, batch.labels)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                predictions = []
                for batch in valid_loader:
                    batch = batch.to(device)
                    pred = model(*model.unpack(batch))
                    loss = loss_fn(pred, batch.labels)
                    pred = pred.detach().cpu().numpy()
                    identifiers = batch.identifiers
                    identifiers["prediction"] = pred
                    predictions.append(identifiers)
                    valid_loss += loss.item()

                predictions = pd.concat(predictions)
                cur_e_perf={
                    name: func(predictions[args.label_name], predictions["prediction"]) if name not in ['f1','acc'] else func(predictions[args.label_name], predictions["prediction"]>0.5) for name, func in metric_dict.items()
                }
                cur_auc = cur_e_perf['auroc']
                if cur_auc > best_auc:
                    best_e = e
                    best_model = copy.deepcopy(model)
                    best_perf = cur_e_perf

        predictions = []
        for batch in test_loader:
            batch = batch.to(device)
            pred = best_model(*model.unpack(batch))
            pred = pred.detach().cpu().numpy()
            identifiers = batch.identifiers
            identifiers["prediction"] = pred
            predictions.append(identifiers)
        predictions = pd.concat(predictions)
        cur_e_perf={
            name: func(predictions[args.label_name], predictions["prediction"]) if name not in ['f1','acc'] else func(predictions[args.label_name], predictions["prediction"]>0.5) for name, func in metric_dict.items()
        }

        performance.loc[i,'auroc'] = cur_e_perf['auroc']
        performance.loc[i,'auprc'] = cur_e_perf['auprc']
        performance.loc[i,'f1'] = cur_e_perf['f1']
        performance.loc[i,'acc'] = cur_e_perf['acc']
       
        perf_file_name = f'{dataname}/CabidaDDI_seed{seeds[i]}_lr{args.lr}_device{args.device}_bestE{best_e}_performance.csv'
        model_file_name = f"{dataname}/CabidaDDI_seed{seeds[i]}_lr{args.lr}_device{args.device}_bestE{best_e}_model.pkl"
        
        if not os.path.exists(perf_file_name):
            performance.to_csv(perf_file_name)
        else:
            performance.to_csv(perf_file_name, mode='a', header=False)
        torch.save(best_model.state_dict(), model_file_name)


