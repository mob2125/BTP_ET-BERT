import torch.nn as nn
from uer.layers.layer_norm import LayerNorm, T5LayerNorm
from uer.layers.position_ffn import PositionwiseFeedForward, GatedFeedForward
from uer.layers.multi_headed_attn import MultiHeadedAttention
from uer.layers.relative_position_embedding import RelativePositionEmbedding


class TransformerLayer(nn.Module):
    """
    Transformer layer mainly consists of two parts:
    multi-headed self-attention and feed forward layer.
    """
    def __init__(self, args):
        super(TransformerLayer, self).__init__()

        self.layernorm_positioning = args.layernorm_positioning

        if hasattr(args, "attention_head_size"):
            attention_head_size = args.attention_head_size
        else:
            attention_head_size = args.hidden_size // args.heads_num

        has_bias = bool(1 - args.remove_transformer_bias)
        with_scale = bool(1 - args.remove_attention_scale)

        # Multi-headed self-attention.
        self.self_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, attention_head_size, args.dropout, has_bias=has_bias, with_scale = with_scale
        )
        self.dropout_1 = nn.Dropout(args.dropout)

        # Feed forward layer.
        if args.feed_forward == "gated":
            self.feed_forward = GatedFeedForward(
                args.hidden_size, args.feedforward_size, args.hidden_act, has_bias
            )
        else:
            self.feed_forward = PositionwiseFeedForward(
                args.hidden_size, args.feedforward_size, args.hidden_act, has_bias
            )
        self.dropout_2 = nn.Dropout(args.dropout)

        if args.layernorm == "t5":
            self.layer_norm_1 = T5LayerNorm(args.hidden_size)
            self.layer_norm_2 = T5LayerNorm(args.hidden_size)
        else:
            self.layer_norm_1 = LayerNorm(args.hidden_size)
            self.layer_norm_2 = LayerNorm(args.hidden_size)



    def forward(self, hidden, mask, position_bias = None):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]
            position_bias: [1 x heads_num x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """

        if self.layernorm_positioning == "post":
            temp = self.self_attn(hidden, hidden, hidden, mask, position_bias)
            inter = self.dropout_1(temp[0])
            inter = self.layer_norm_1(inter + hidden)
            output = self.dropout_2(self.feed_forward(inter))
            output = self.layer_norm_2(output + inter)
        else:
            inter = self.layer_norm_1(hidden)
            temp = self.self_attn(inter, inter, inter, mask, position_bias)
            inter = self.dropout_1(temp[0])
            hidden = hidden + inter
            output = self.layer_norm_2(hidden)
            output = self.dropout_2(self.feed_forward(output)) + hidden
        return output, temp[1]


class TransformerDecoderLayer(nn.Module):
    def __init__(self, args):
        super(TransformerDecoderLayer, self).__init__()

        self.layernorm_positioning = args.layernorm_positioning

        if hasattr(args, "attention_head_size"):
            attention_head_size = args.attention_head_size
        else:
            attention_head_size = args.hidden_size // args.heads_num

        has_bias = bool(1 - args.remove_transformer_bias)
        with_scale = bool(1 - args.remove_attention_scale)

        # Multi-headed self-attention.
        self.self_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, attention_head_size, args.dropout, has_bias=has_bias, with_scale = with_scale
        )
        self.dropout_1 = nn.Dropout(args.dropout)

        # Multi-headed context-attention.
        self.context_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, attention_head_size, args.dropout, has_bias=has_bias, with_scale = with_scale
        )
        self.dropout_2 = nn.Dropout(args.dropout)

        # Feed forward layer.
        if args.feed_forward == "gated":
            self.feed_forward = GatedFeedForward(
                args.hidden_size, args.feedforward_size, args.hidden_act, has_bias
            )
        else:
            self.feed_forward = PositionwiseFeedForward(
                args.hidden_size, args.feedforward_size, args.hidden_act, has_bias
            )
        self.dropout_3 = nn.Dropout(args.dropout)

        # Layer Normalization
        if  args.layernorm == "t5":
            self.layer_norm_1 = T5LayerNorm(args.hidden_size)
            self.layer_norm_2 = T5LayerNorm(args.hidden_size)
            self.layer_norm_3 = T5LayerNorm(args.hidden_size)
        else:
            self.layer_norm_1 = LayerNorm(args.hidden_size)
            self.layer_norm_2 = LayerNorm(args.hidden_size)
            self.layer_norm_3 = LayerNorm(args.hidden_size)



    def forward(self, hidden, encoder_hidden, mask_decoder, mask_encoder, self_position_bias = None, context_position_bias = None):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            encoder_hidden: [batch_size x seq_length x emb_size]
            mask_encoder: [batch_size x 1 x seq_length x seq_length]
            mask_decoder: [batch_size x 1 x seq_length x seq_length]
            self_position_bias: [1 x heads_num x seq_length x seq_length]
            context_position_bias: [1 x heads_num x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """

        if self.layernorm_positioning == "post":
            query = self.dropout_1(self.self_attn(hidden, hidden, hidden, mask_decoder, self_position_bias))
            query_norm = self.layer_norm_1(query + hidden)
            mid = self.dropout_2(self.context_attn(encoder_hidden, encoder_hidden, query_norm, mask_encoder, context_position_bias))
            mid_norm = self.layer_norm_2(mid + query_norm)
            output = self.dropout_3(self.feed_forward(mid_norm))
            output = self.layer_norm_3(output + mid_norm)
        else:
            hidden_norm = self.layer_norm_1(hidden)
            query = self.dropout_1(self.self_attn(hidden_norm, hidden_norm, hidden_norm, mask_decoder, self_position_bias))
            query = query + hidden
            query_norm = self.layer_norm_2(query)
            mid = self.dropout_2(self.context_attn(encoder_hidden, encoder_hidden, query_norm, mask_encoder, context_position_bias))
            mid = mid + query
            mid_norm = self.layer_norm_3(mid)
            output = self.dropout_3(self.feed_forward(mid_norm)) + mid
        return output


#class GptBlock(nn.Module):
#    def __init__(self, args):
#        super(GptBlock, self).__init__()

#        # Multi-headed self-attention.
#        self.self_attn = MultiHeadedAttention(
#            args.hidden_size, args.heads_num, args.dropout
#        )
#        self.layer_norm_1 = LayerNorm(args.hidden_size)
#        # Feed forward layer.
#        self.feed_forward = PositionwiseFeedForward(
#            args.hidden_size, args.feedforward_size, args.hidden_act
#        )
#        self.layer_norm_2 = LayerNorm(args.hidden_size)

#    def forward(self, hidden, mask):
#        """
#        Args:
#            hidden: [batch_size x seq_length x emb_size]
#            mask: [batch_size x 1 x seq_length x seq_length]
#        Returns:
#            output: [batch_size x seq_length x hidden_size]
#        """
#        inter = self.layer_norm_1(hidden)
#        inter = self.self_attn(inter, inter, inter, mask)
#        hidden = hidden + inter
#        output = self.layer_norm_2(hidden)
#        output = self.feed_forward(output)
        
#        return output + hidden
