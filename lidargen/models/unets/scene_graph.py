import torch
import torch.nn as nn
import numpy as np
from .graph import GraphTripleConvNet

class SceneGraph(nn.Module):
    def __init__(self, vocab, embedding_dim=128, batch_size=32,
                 gconv_pooling='avg', gconv_num_layers=5,
                 mlp_normalization='none',
                 separated=False,
                 replace_latent=False,
                 residual=False,
                 use_angles=False,
                 use_clip=True):
        super(SceneGraph, self).__init__()

        gconv_dim = embedding_dim
        gconv_hidden_dim = gconv_dim * 4
        self.replace_all_latent = replace_latent
        self.batch_size = batch_size
        self.embedding_dim = gconv_dim
        self.vocab = vocab
        self.use_angles = use_angles
        self.clip = use_clip
        add_dim = 0
        if self.clip:
            add_dim = 512
        self.edge_list = list(set(vocab['pred_idx_to_name']))
        self.obj_classes_list = list(set(vocab['object_idx_to_name']))
        self.classes = dict(zip(sorted(self.obj_classes_list),range(len(self.obj_classes_list))))
        self.classes_r = dict(zip(self.classes.values(), self.classes.keys()))
        num_objs = len(self.obj_classes_list)
        num_preds = len(self.edge_list)

        # build graph encoder and manipulator
        self.obj_embeddings_ec = nn.Embedding(num_objs + 1, gconv_dim * 2)
        self.pred_embeddings_ec = nn.Embedding(num_preds, gconv_dim * 2)
        self.obj_embeddings_dc = nn.Embedding(num_objs + 1, gconv_dim * 2) # TODO is this necessary?
        self.pred_embeddings_man_dc = nn.Embedding(num_preds, gconv_dim * 2)

        self.out_dim_ini_encoder = gconv_dim * 2 + add_dim
        gconv_kwargs_ec = {
            'input_dim_obj': gconv_dim * 2 + add_dim,
            'input_dim_pred': gconv_dim * 2 + add_dim,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'num_layers': gconv_num_layers,
            'mlp_normalization': mlp_normalization,
            'residual': residual,
            'output_dim': self.out_dim_ini_encoder
        }
        self.out_dim_manipulator = gconv_dim * 2 + add_dim
        gconv_kwargs_manipulation = {
            'input_dim_obj': self.out_dim_ini_encoder + gconv_dim + gconv_dim * 2 + add_dim, # latent_f + change_flag + obj_embedding + clip
            'input_dim_pred': gconv_dim * 2 + add_dim,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'num_layers': min(gconv_num_layers, 5),
            'mlp_normalization': mlp_normalization,
            'residual': residual,
            'output_dim': self.out_dim_manipulator
        }
        self.gconv_net_ec = GraphTripleConvNet(**gconv_kwargs_ec)
        self.gconv_net_manipulation = GraphTripleConvNet(**gconv_kwargs_manipulation)

        self.s_l_separated = separated
        if self.s_l_separated:
            gconv_kwargs_ec_rel = {
                'input_dim_obj': self.out_dim_manipulator + gconv_dim * 2 + add_dim,
                'input_dim_pred': gconv_dim * 2 + add_dim,
                'hidden_dim': gconv_hidden_dim,
                'pooling': gconv_pooling,
                'num_layers': gconv_num_layers,
                'mlp_normalization': mlp_normalization,
                'residual': residual,
                'output_dim': self.out_dim_manipulator
            }
            self.gconv_net_ec_rel_l = GraphTripleConvNet(**gconv_kwargs_ec_rel)

    def init_encoder(self, objs, triples, enc_text_feat, enc_rel_feat):
        O, T = objs.size(0), triples.size(0)
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_embed = self.obj_embeddings_ec(objs)
        pred_embed = self.pred_embeddings_ec(p)
        if self.clip:
            obj_embed = torch.cat([enc_text_feat, obj_embed], dim=1)
            pred_embed = torch.cat([enc_rel_feat, pred_embed], dim=1)

        latent_obj_f, latent_pred_f = self.gconv_net_ec(obj_embed, pred_embed, edges)

        return obj_embed, pred_embed, latent_obj_f, latent_pred_f

    def manipulate(self, latent_f, objs, triples, dec_text_feat, dec_rel_feat):
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_embed = self.obj_embeddings_ec(objs) # TODO is obj_embeddings_ec enough here?
        pred_embed = self.pred_embeddings_man_dc(p)
        if self.clip:
            obj_embed = torch.cat([dec_text_feat, obj_embed], dim=1)
            pred_embed = torch.cat([dec_rel_feat, pred_embed], dim=1)

        obj_vecs_ = torch.cat([latent_f, obj_embed], dim=1)
        obj_vecs_, pred_vecs_ = self.gconv_net_manipulation(obj_vecs_, pred_embed, edges)

        return obj_vecs_, pred_vecs_, obj_embed, pred_embed

    def forward(self, enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat,\
                    dec_objs, dec_triples, dec_boxes,\
                    encoded_dec_text_feat, encoded_dec_rel_feat, dec_objs_to_scene,\
                    dec_triples_to_scene, missing_nodes, manipulated_nodes):
        obj_embed, pred_embed, latent_obj_vecs, latent_pred_vecs = self.init_encoder(enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat)

        # append zero nodes
        nodes_added = []
        for i in range(len(missing_nodes)):
          ad_id = missing_nodes[i] + i
          nodes_added.append(ad_id)
          noise = np.zeros(self.out_dim_ini_encoder)
          zeros = torch.from_numpy(noise.reshape(1, self.out_dim_ini_encoder))
          zeros.requires_grad = True
          zeros = zeros.float().cuda()
          latent_obj_vecs = torch.cat([latent_obj_vecs[:ad_id], zeros, latent_obj_vecs[ad_id:]], dim=0)

        # mark changes in nodes
        change_repr = []
        for i in range(len(latent_obj_vecs)):
            if i not in nodes_added and i not in manipulated_nodes:
                noisechange = np.zeros(self.embedding_dim)
            else:
                noisechange = np.random.normal(0, 1, self.embedding_dim)
            change_repr.append(torch.from_numpy(noisechange).float().cuda())
        change_repr = torch.stack(change_repr, dim=0)
        latent_obj_vecs_ = torch.cat([latent_obj_vecs, change_repr], dim=1)
        latent_obj_vecs_, _, obj_embed_, _ = self.manipulate(latent_obj_vecs_, dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat) # contains all obj now

        if not self.replace_all_latent:
            # take original nodes when untouched
            touched_nodes = torch.tensor(sorted(nodes_added + manipulated_nodes)).long()
            for touched_node in touched_nodes:
                latent_obj_vecs = torch.cat([latent_obj_vecs[:touched_node], latent_obj_vecs_[touched_node:touched_node + 1], latent_obj_vecs[touched_node + 1:]], dim=0)
        else:
            latent_obj_vecs = latent_obj_vecs_

        return latent_obj_vecs, obj_embed_