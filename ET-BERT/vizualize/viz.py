import sys
import os
import torch
import argparse
import collections
import torch.nn as nn

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.utils.constants import *
from uer.utils.vocab import Vocab
from uer.utils.optimizers import *
from uer.utils import *
from uer.layers import *
from uer.encoders import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_loader import load_model
from uer.opts import infer_opts
from bertviz import model_view

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def forward(self, src, tgt, seg, soft_tgt=None):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output, attn_scores = self.encoder(emb, seg)
        temp_output = output
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                       (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            else:
                loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits, attn_scores
        else:
            return None, logits, attn_scores
            #return temp_output, logits

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    parser.add_argument("--labels_num", type=int, required=True,
                        help="Number of prediction labels.")

    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    parser.add_argument("--output_logits", action="store_true", help="Write logits to output file.")
    parser.add_argument("--output_prob", action="store_true", help="Write probabilities to output file.")
    parser.add_argument("--html_file", type=str, default="attention.html", help="The name of the output html file.")
    parser.add_argument("--include_layers", type=str, default="-1", help="The layers to include, e.g. \"-1, -2, -3\".")
    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)
    layers_to_include = args.include_layers.split(",")
    layers_to_include = [int(x) for x in layers_to_include]
    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model and load parameters.
    args.soft_targets, args.soft_alpha = False, False

    model = Classifier(args)
    model = load_model(model, args.load_model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    input_string = "3d45 4500 001a 1ad3 d3ae ae46 4685 8502 02be be0f 0fb3 b1a7 a6ac ac4c 4c43 4326 2625 257b 7b5c 5c9f 9f01 0162 6254"
    input_ids = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(input_string))
    segment_ids = [1] * len(input_ids)
    if len(input_ids) > args.seq_length:
        input_ids = input_ids[: args.seq_length]
        segment_ids = segment_ids[: args.seq_length]
    while len(input_ids) < args.seq_length:
        input_ids.append(0)
        segment_ids.append(0)
    input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)
    segment_ids = torch.LongTensor(segment_ids).unsqueeze(0).to(device)

    output = model(input_ids, None, segment_ids)
    attention = output[-1]
    logits = output[-2]
    pred = torch.argmax(logits, dim=1)
    pred = pred.cpu().numpy().tolist()
    prob = nn.Softmax(dim=1)(logits)
    for j in range(len(pred)):
        print("Pred_value:", str(pred[j]))
    
    tokens = args.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    ht = model_view(attention, tokens, html_action='return', include_layers=layers_to_include)
    print(type(ht))
    with open(args.html_file, 'w', encoding='utf-8') as f:
        f.write(ht.data)
    print("HTML content saved to vizualize.html")

if __name__== "__main__":
    main()

    

