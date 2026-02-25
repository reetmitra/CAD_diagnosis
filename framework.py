import os
import random
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader

from architecture import spatio_temporal_semantic_learning
from config import opt
import functions as funcs
import optimization as opt_fn
import augmentation as aug


class sc_net_framework:

    def __init__(self, pattern='pre_training', state_dict_root=None, data_root=None,
                 delta=1.0, sc_class_weights=None,
                 use_focal=False, focal_gamma=2.0,
                 temporal_encoder_layers=None, temporal_heads=None,
                 spatial_encoder_layers=None, spatial_decoder_layers=None):

        if pattern == "pre_training":
            self.model_pattern = "training"
            self.model_num_classes = opt.net_params["num_classes"][0]
        elif pattern == "fine_tuning":
            self.model_pattern = "training"
            self.model_num_classes = opt.net_params["num_classes"][1]
        else:
            self.model_pattern = "testing"
            self.model_num_classes = opt.net_params["num_classes"][1]

        # Store focal loss settings
        self._use_focal = use_focal
        self._focal_gamma = focal_gamma

        # Store transformer overrides
        self._temporal_encoder_layers = temporal_encoder_layers
        self._temporal_heads = temporal_heads
        self._spatial_encoder_layers = spatial_encoder_layers
        self._spatial_decoder_layers = spatial_decoder_layers

        self.model = self.get_model()
        self.state_dict_root = state_dict_root
        if self.state_dict_root is not None:
            self.pre_training_load()

        if pattern != 'inference':

            if data_root is not None:
                self.data_root = data_root
            elif pattern == 'pre_training':
                self.data_root = opt.data_params["pretrain_data_root"]
            elif pattern == 'fine_tuning':
                self.data_root = opt.data_params["finetune_data_root"]
            else:
                self.data_root = opt.data_params["dataset_root"]

            self.train_ratio = opt.data_params["train_ratio"]
            self.input_shape = opt.net_params["input_shape"]
            self.window_lw = opt.data_params["window_lw"]
            self.batch_size = opt.data_params["batch_size"]

            self.loss_fn = self.get_loss_fn(delta=delta, sc_class_weights=sc_class_weights,
                                               use_focal=self._use_focal,
                                               focal_gamma=self._focal_gamma)
            self.dataLoader_train, self.dataLoader_eval, self.dataLoader_test = self.get_dataloader()

    def get_model(self):
        # Build transformer param lists, applying any overrides
        # temporal_transfromer_param = [heads, encoder_layers]
        temporal_tf = list(opt.sc_params["temporal_transfromer_param"])
        if self._temporal_heads is not None:
            temporal_tf[0] = self._temporal_heads
        if self._temporal_encoder_layers is not None:
            temporal_tf[1] = self._temporal_encoder_layers

        # spatial_transfromer_param = [encoder_layers, decoder_layers]
        spatial_tf = list(opt.od_params["spatial_transfromer_param"])
        if self._spatial_encoder_layers is not None:
            spatial_tf[0] = self._spatial_encoder_layers
        if self._spatial_decoder_layers is not None:
            spatial_tf[1] = self._spatial_decoder_layers

        return spatio_temporal_semantic_learning(
            num_classes=self.model_num_classes,
            pattern=self.model_pattern,
            ret_map=opt.net_params["ret_map"],
            in_channels=opt.net_params["in_channels"],
            _3d_cube_selection=opt.sc_params["_3d_cube_selection"],
            temporal_conv_levels=opt.sc_params["temporal_conv_levels"],
            temporal_conv_maps=opt.sc_params["temporal_conv_maps"],
            temporal_feature_channels=opt.sc_params["temporal_feature_channels"],
            temporal_embedding_dim=opt.sc_params["temporal_embedding_dim"],
            temporal_transfromer_param=temporal_tf,
            temporal_class_dim=opt.sc_params["temporal_class_dim"],
            spatial_conv_levels=opt.od_params["spatial_conv_levels"],
            spatial_conv_maps=opt.od_params["spatial_conv_maps"],
            spatial_3dconv_layers=opt.od_params["spatial_3dconv_layers"],
            spatial_2dconv_layers=opt.od_params["spatial_2dconv_layers"],
            spatial_2d_weight=opt.od_params["spatial_2d_weight"],
            spatial_3d_weight=opt.od_params["spatial_3d_weight"],
            spatial_proj_channels=opt.od_params["spatial_proj_channels"],
            spatial_embedding_shape=opt.od_params["spatial_embedding_shape"],
            spatial_transfromer_param=spatial_tf,
            spatial_num_query=opt.od_params["spatial_num_query"],
            spatial_od_dim_list=opt.od_params["spatial_od_dim_list"]
        )

    def get_loss_fn(self, delta=1.0, sc_class_weights=None,
                    use_focal=False, focal_gamma=2.0):
        return opt_fn.spatio_temporal_contrast_loss(
            num_classes=self.model_num_classes,
            seq_length=opt.net_params["cubeseq_length"],
            eos_coef=opt.data_params["eos_coef"],
            delta=delta,
            sc_class_weights=sc_class_weights,
            use_focal=use_focal,
            focal_gamma=focal_gamma,
        )

    def get_dataloader(self):
        dataset_training = aug.cubic_sequence_data(
            dataset_root=self.data_root,
            pattern='training',
            train_ratio=self.train_ratio,
            input_shape=self.input_shape,
            window=self.window_lw,
            num_classes=self.model_num_classes)
        dataset_validation = aug.cubic_sequence_data(
            dataset_root=self.data_root,
            pattern='validation',
            train_ratio=self.train_ratio,
            input_shape=self.input_shape,
            window=self.window_lw,
            num_classes=self.model_num_classes)
        dataset_testing = aug.cubic_sequence_data(
            dataset_root=self.data_root,
            pattern='testing',
            train_ratio=self.train_ratio,
            input_shape=self.input_shape,
            window=self.window_lw,
            num_classes=self.model_num_classes)
        return DataLoader(dataset_training, batch_size=self.batch_size, shuffle=True, collate_fn=aug.collate_fn),\
               DataLoader(dataset_validation, batch_size=self.batch_size, shuffle=False, collate_fn=aug.collate_fn),\
               DataLoader(dataset_testing, batch_size=self.batch_size, shuffle=False, collate_fn=aug.collate_fn)

    def pre_training_load(self):

        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(self.state_dict_root, map_location='cpu',
                                     weights_only=False)

        # Handle checkpoints saved by Trainer (wrapped under 'model_state_dict')
        if 'model_state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['model_state_dict']

        pretrained_dict_filtered = {}
        skipped = []
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    pretrained_dict_filtered[k] = v
                else:
                    skipped.append(f"{k}: pretrained {list(v.shape)} vs model {list(model_dict[k].shape)}")

        model_dict.update(pretrained_dict_filtered)
        self.model.load_state_dict(model_dict)

        print(f"[pre_training_load] Loaded {len(pretrained_dict_filtered)}/{len(model_dict)} parameters")
        if skipped:
            print(f"[pre_training_load] Skipped {len(skipped)} mismatched layers:")
            for s in skipped:
                print(f"  - {s}")
