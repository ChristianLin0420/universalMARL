import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

class SelfAttention(nn.Module):
    def __init__(self, args, emb, heads=8, mask=False, cross=False):

        super().__init__()

        self.args = args
        self.emb = emb
        self.heads = heads
        self.mask = mask

        key_in  = emb // 2 if args.agent == "trackformer" and cross else emb
        query_in = emb 
        value_in = key_in

        self.tokeys = nn.Linear(key_in, emb * heads, bias=False)
        self.toqueries = nn.Linear(query_in, emb * heads, bias=False)
        self.tovalues = nn.Linear(value_in, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x, mask, encoder_output = None, save_attention_maps=False, save_path=None, layer_idx=None, frame_idx=None):

        b, t, e = x.size()
        h = self.heads

        if self.args.agent == "trackformer":
            e = self.emb

        if encoder_output is not None:
            bq, tq, eq = encoder_output.size()
            keys = self.tokeys(x).view(b, t, h, e)
            queries = self.toqueries(encoder_output).view(b, tq, h, e)
            values = self.tovalues(x).view(b, t, h, e)
        else:
            tq = t
            keys = self.tokeys(x).view(b, t, h, e)
            queries = self.toqueries(x).view(b, tq, h, e)
            values = self.tovalues(x).view(b, t, h, e)

        # compute scaled dot-product self-attention
        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, tq, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b * h, tq, t)

        if self.mask:  # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        if mask is not None:
            dot = dot.masked_fill(mask == 0, -1e9)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        if save_attention_maps and save_path is not None:
            attention_weights = dot.view(b, h, tq, t)[0, :, :, :].detach().cpu().numpy()
            
            # Create directories for images and videos
            layer_path = Path(save_path) / f'layer_{layer_idx}'
            layer_path.mkdir(parents=True, exist_ok=True)
            
            (layer_path / 'images').mkdir(exist_ok=True)
            (layer_path / 'videos').mkdir(exist_ok=True)
            
            # Save individual frame heatmaps
            for head_idx in range(h):
                attn = attention_weights[head_idx, :, :]
                
                # Create heatmap image
                plt.figure(figsize=(10, 8))
                sns.heatmap(attn, cmap='viridis', annot=True, fmt='.2f')
                plt.title(f'Layer {layer_idx}, Head {head_idx}, Frame {frame_idx}')
                plt.xlabel('Key sequence')
                plt.ylabel('Query sequence')
                
                # Save image
                img_path = layer_path / 'images' / f'head_{head_idx}_frame_{frame_idx}.png'
                plt.savefig(img_path)
                plt.close()
                
                # Update video file
                video_path = layer_path / 'videos' / f'head_{head_idx}_attention.mp4'
                
                # If this is the first frame, initialize the video writer
                if frame_idx == 0:
                    self._initialize_video_writer(video_path, attn.shape)
                
                # Convert heatmap to BGR format for video
                fig = plt.figure(figsize=(10, 8))
                sns.heatmap(attn, cmap='viridis', annot=True, fmt='.2f')
                plt.title(f'Layer {layer_idx}, Head {head_idx}')
                plt.xlabel('Key sequence')
                plt.ylabel('Query sequence')
                
                # Convert plot to image array
                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Write frame to video
                if hasattr(self, f'video_writer_head_{head_idx}'):
                    getattr(self, f'video_writer_head_{head_idx}').write(frame)
                
                plt.close()

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, tq, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, tq, h * e)

        return self.unifyheads(out)

    def _initialize_video_writer(self, video_path, attn_shape):
        """Initialize video writer for each attention head."""
        # Create a temporary figure to get the frame size
        fig = plt.figure(figsize=(10, 8))
        plt.close()
        
        # Get frame size from figure
        dpi = fig.get_dpi()
        width = int(10 * dpi)
        height = int(8 * dpi)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Extract head index from video path
        head_idx = int(str(video_path).split('head_')[1].split('_')[0])
        
        # Create video writer and store it as an instance attribute
        video_writer = cv2.VideoWriter(
            str(video_path), 
            fourcc, 
            5.0,  # fps
            (width, height)
        )
        setattr(self, f'video_writer_head_{head_idx}', video_writer)

    def __del__(self):
        """Clean up video writers when the object is destroyed."""
        for attr in dir(self):
            if attr.startswith('video_writer_head_'):
                getattr(self, attr).release()

def mask_(matrices, maskval=0.0, mask_diagonal=True):

    b, h, w = matrices.size()
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval