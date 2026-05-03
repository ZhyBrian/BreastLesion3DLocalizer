import torch
from torch import nn

from argparse import Namespace

from spatial_encoder.Conformer_Unet_MTL_mtr_base_test_out768 import ConvFormer_MTL as ConvFormer_MTL_test
from spatial_encoder.Conformer_Unet_MTL_mtr_base_out768 import ConvFormer_MTL as ConvFormer_MTL_train

import hlst_helper as hlst_helper


class HLST(nn.Module):
    def __init__(self, *, num_classes, spatial_frozen, spatial_args, temporal_type, temporal_args,
                 temporal_pretrained_weights=None,
                 mlp_head_weights=None,
                 cls_token_weights=None):
        super().__init__()
        # self.frames = frames

        # Convert args
        spatial_args = Namespace(**spatial_args)
        temporal_args = Namespace(**temporal_args)

        # self.collapse_frames = Rearrange('b f c h w -> (b f) c h w')

        #[Spatial] Transformer attention 
        if spatial_frozen:
            self.spatial_transformer = ConvFormer_MTL_test()
        else:
            self.spatial_transformer = ConvFormer_MTL_train()
        self.spatial_transformer.load_state_dict(torch.load(spatial_args.pretrained_weights))
        
        # Freeze spatial backbone
        self.spatial_frozen = spatial_frozen
        if spatial_frozen:
            self.spatial_transformer.eval()

        #[Temporal] Transformer_attention
        assert temporal_type in ['longformer', 'linformer', 'transformer'], "Only longformer, linformer, transformer are supported"
        # Copy seq_len to frames
        # temporal_args.seq_len = frames
        self.temporal_type = temporal_type
        
        if temporal_type == 'longformer':
            self.cls_token = nn.Parameter(torch.randn(1, 1, temporal_args.EMBED_DIM))
            if cls_token_weights is not None:
                self.cls_token.data.copy_(torch.load(cls_token_weights)['cls_token'])
                assert torch.allclose(self.cls_token, torch.load(cls_token_weights)['cls_token'], atol=1e-6)
                assert self.cls_token.shape == torch.load(cls_token_weights)['cls_token'].shape
            # self.temporal_transformer = Longformer(**vars(temporal_args))
            self.temporal_transformer = hlst_helper.HLSTLongformerModel(
                                        embed_dim=temporal_args.EMBED_DIM,
                                        max_position_embeddings=temporal_args.MAX_POSITION_EMBEDDINGS,
                                        num_attention_heads=temporal_args.NUM_ATTENTION_HEADS,
                                        num_hidden_layers=temporal_args.NUM_HIDDEN_LAYERS,
                                        attention_mode=temporal_args.ATTENTION_MODE,
                                        pad_token_id=temporal_args.PAD_TOKEN_ID,
                                        attention_window=temporal_args.ATTENTION_WINDOW,
                                        intermediate_size=temporal_args.INTERMEDIATE_SIZE,
                                        attention_probs_dropout_prob=temporal_args.ATTENTION_PROBS_DROPOUT_PROB,
                                        hidden_dropout_prob=temporal_args.HIDDEN_DROPOUT_PROB)
            if temporal_pretrained_weights is not None:
                missing, unexpected = self.temporal_transformer.load_state_dict(torch.load(temporal_pretrained_weights), strict=False)
                print(f"Missing keys temporal_transformer: {missing}")
                print(f"Unexpected keys temporal_transformer: {unexpected}")
        elif temporal_type == 'linformer':
            raise NotImplementedError("Linformer is not implemented yet")
        elif temporal_type == 'transformer':
            raise NotImplementedError("Transformer is not implemented yet")

        # Classifer
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(temporal_args.HIDDEN_DIM),
            nn.Linear(temporal_args.HIDDEN_DIM, temporal_args.MLP_DIM),
            nn.GELU(),
            nn.Dropout(temporal_args.DROPOUT_RATE),
            nn.Linear(temporal_args.MLP_DIM, num_classes)
        )
        # Random init 0.0 mean, 0.02 std
        nn.init.normal_(self.mlp_head[1].weight, mean=0.0, std=0.02)
        nn.init.normal_(self.mlp_head[4].weight, mean=0.0, std=0.02)
        
        if mlp_head_weights is not None:
            missing, unexpected = self.mlp_head.load_state_dict(torch.load(mlp_head_weights), strict=False)
            print(f"Missing keys mlp_head: {missing}")
            print(f"Unexpected keys mlp_head: {unexpected}")


    def forward(self, video, masks=None, position_ids=None):

        # x = self.collapse_frames(video)
        B, C, F, H, W = video.shape
        x = video.permute(0, 2, 1, 3, 4)
        x = x.reshape(B * F, C, H, W)
        if masks is not None:
            masks = masks.permute(0, 2, 1, 3, 4)
            masks = masks.reshape(B * F, 1, H, W)
        
        # Spatial Transformer
        if self.spatial_frozen:
            with torch.no_grad():
                if masks is not None:
                    _, _, x = self.spatial_transformer(x, masks)
                else:
                    _, _, x = self.spatial_transformer(x)
        else:
            if masks is not None:
                out_f_mask, out_f_cls, x = self.spatial_transformer(x, masks)
            else:
                out_f_mask, out_f_cls, x = self.spatial_transformer(x)
  
        # Spatial to temporal
        # x = self.spatial2temporal(x)
        x = x.reshape(B, F, -1)
        if position_ids is None:
            position_ids = torch.arange(F, device=x.device).expand((B, F))

        # Temporal Transformer
        # x = self.temporal_transformer(x)
        B, D, E = x.shape
        attention_mask = torch.ones((B, D), dtype=torch.long, device=x.device)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        cls_atten = torch.ones(1).expand(B, -1).to(x.device)
        attention_mask = torch.cat((attention_mask, cls_atten), dim=1)
        attention_mask[:, 0] = 2
        # print("x.shape, ", x.shape)
        # print("attention_mask, ", attention_mask)
        x, attention_mask, position_ids = hlst_helper.pad_to_window_size_local(
            x,
            attention_mask,
            position_ids,
            self.temporal_transformer.config.attention_window[0],
            self.temporal_transformer.config.pad_token_id)
        # print("x.shape, ", x.shape)
        # print("attention_mask, ", attention_mask)
        # print("position_ids, ", position_ids)
        token_type_ids = torch.zeros(x.size()[:-1], dtype=torch.long, device=x.device)
        token_type_ids[:, 0] = 1
        # print("token_type_ids, ", token_type_ids)

        # position_ids
        position_ids = position_ids.long()
        mask = attention_mask.ne(0).int()
        max_position_embeddings = self.temporal_transformer.config.max_position_embeddings
        position_ids = position_ids % (max_position_embeddings - 2)
        position_ids[:, 0] = max_position_embeddings - 2
        position_ids[mask == 0] = max_position_embeddings - 1
        # print("position_ids, ", position_ids)

        x = self.temporal_transformer(input_ids=None,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        inputs_embeds=x,
                                        output_attentions=None,
                                        output_hidden_states=None,
                                        return_dict=None)
        
        # MLP head
        x = x["last_hidden_state"]
        x = self.mlp_head(x[:, 0])
        
        if self.spatial_frozen:
            # If spatial frozen, return only the prediction
            return x
        else:   
            # If not frozen, return both the prediction and the spatial features
            return x, out_f_mask, out_f_cls
    
    
    def forward_att(self, video, masks=None, position_ids=None):

        # x = self.collapse_frames(video)
        B, C, F, H, W = video.shape
        x = video.permute(0, 2, 1, 3, 4)
        x = x.reshape(B * F, C, H, W)
        if masks is not None:
            masks = masks.permute(0, 2, 1, 3, 4)
            masks = masks.reshape(B * F, 1, H, W)
        
        # Spatial Transformer
        if self.spatial_frozen:
            with torch.no_grad():
                if masks is not None:
                    _, _, x = self.spatial_transformer(x, masks)
                else:
                    _, _, x = self.spatial_transformer(x)
        else:
            if masks is not None:
                out_f_mask, out_f_cls, x = self.spatial_transformer(x, masks)
            else:
                out_f_mask, out_f_cls, x = self.spatial_transformer(x)
  
        # Spatial to temporal
        # x = self.spatial2temporal(x)
        x = x.reshape(B, F, -1)
        if position_ids is None:
            position_ids = torch.arange(F, device=x.device).expand((B, F))

        # Temporal Transformer
        # x = self.temporal_transformer(x)
        B, D, E = x.shape
        attention_mask = torch.ones((B, D), dtype=torch.long, device=x.device)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        cls_atten = torch.ones(1).expand(B, -1).to(x.device)
        attention_mask = torch.cat((attention_mask, cls_atten), dim=1)
        attention_mask[:, 0] = 2
        # print("x.shape, ", x.shape)
        # print("attention_mask, ", attention_mask)
        x, attention_mask, position_ids = hlst_helper.pad_to_window_size_local(
            x,
            attention_mask,
            position_ids,
            self.temporal_transformer.config.attention_window[0],
            self.temporal_transformer.config.pad_token_id)
        # print("x.shape, ", x.shape)
        # print("attention_mask, ", attention_mask)
        # print("position_ids, ", position_ids)
        token_type_ids = torch.zeros(x.size()[:-1], dtype=torch.long, device=x.device)
        token_type_ids[:, 0] = 1
        # print("token_type_ids, ", token_type_ids)

        # position_ids
        position_ids = position_ids.long()
        mask = attention_mask.ne(0).int()
        max_position_embeddings = self.temporal_transformer.config.max_position_embeddings
        position_ids = position_ids % (max_position_embeddings - 2)
        position_ids[:, 0] = max_position_embeddings - 2
        position_ids[mask == 0] = max_position_embeddings - 1
        # print("position_ids, ", position_ids)

        x = self.temporal_transformer(input_ids=None,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        inputs_embeds=x,
                                        output_attentions=True,
                                        output_hidden_states=None,
                                        return_dict=True)
        
        # print(x.keys())
        
        return x.attentions, x.global_attentions, F

