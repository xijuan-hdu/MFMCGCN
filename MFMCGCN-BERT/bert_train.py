# encoding=utf-8
import sys
sys.path.append('../')
import torch
import random
import argparse
import numpy as np
from vocab import Vocab
from utils import helper
import os
from sklearn import metrics
from bert_loader import ABSADataLoader
from bert_trainer import ABSATrainer
from tensorboardX import SummaryWriter
from transformers import BertTokenizer
logger = SummaryWriter(log_dir="data/log")
writer = SummaryWriter('data/log')

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="dataset/Restaurants")
parser.add_argument("--vocab_dir", type=str, default="dataset/Restaurants")
parser.add_argument("--emb_dim", type=int, default=768, help="bert dim.")
parser.add_argument("--hidden_dim", type=int, default=300, help="GCN mem dim.")
parser.add_argument("--dep_dim", type=int, default=30, help="dep embedding dimension.")
parser.add_argument("--pos_dim", type=int, default=30, help="pos embedding dimension.")
parser.add_argument("--post_dim", type=int, default=30, help="position embedding dimension.")
parser.add_argument("--num_class", type=int, default=3, help="Num of sentiment class.")
parser.add_argument('--rnn_hidden', type=int, default=300, help='RNN hidden state size.')
parser.add_argument("--rnn_layers", type=int, default=1, help="Number of RNN layers.")
parser.add_argument("--input_dropout", type=float, default=0.1, help="Input dropout rate.")
parser.add_argument("--layer_dropout", type=float, default=0, help="RGAT layer dropout rate.")
parser.add_argument(
    "--att_dropout", type=float, default=0, help="self-attention layer dropout rate."
)
parser.add_argument('--rnn_dropout', type=float, default=0.3)
parser.add_argument("--lower", default=True, help="Lowercase all words.")
parser.add_argument("--head_num_GCN", type=int,default=2)
parser.add_argument("--attn_heads", type=int, default=5, help="Num of GAT/RGAT attention heads.")
parser.add_argument("--direct", default=False)
parser.add_argument("--loop", default=True)
parser.add_argument("--reset_pooling", default=False, action="store_true")
parser.add_argument("--lr", type=float, default=2e-5, help="learning rate.")
parser.add_argument("--bidirect", default=True, help="Do use bi-RNN layer.")
parser.add_argument("--num_layers_GCN", type=int, default=2, help='Num of GCN layers.')
parser.add_argument("--bert_lr", type=float, default=2e-5, help="learning rate for bert.")
parser.add_argument('--gcn_dropout', type=float, default=0.4, help='GCN layer dropout rate.')
parser.add_argument("--l2", type=float, default=1e-5, help="weight decay rate.")
parser.add_argument(
    "--optim",
    choices=["sgd", "adagrad", "adam", "adamax"],
    default="adam",
    help="Optimizer: sgd, adagrad, adam or adamax.",
)
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument("--num_layer", type=int, default=3, help="Number of graph layers.")
parser.add_argument("--num_epoch", type=int, default=20, help="Number of total training epochs.")
parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
parser.add_argument("--log_step", type=int, default=16, help="Print log every k steps.")
parser.add_argument(
    "--save_dir", type=str, default="./saved_models/res14", help="Root dir for saving models."
)
parser.add_argument("--model", type=str, default="RGAT", help="model to use, (std, GAT, RGAT)")
parser.add_argument("--seed", type=int, default=29)
parser.add_argument("--bert_out_dim", type=int, default=100)
parser.add_argument(
    "--output_merge",
    type=str,
    default="gatenorm2",
    help="merge method to use, (none, addnorm, add, attn, gate, gatenorm2)",
)
parser.add_argument('--second_layer', type=str, default='max')
parser.add_argument("--max_len", type=int, default=80)
parser.add_argument("--device", default='cuda:0')
parser.add_argument('--top_k', default=2, type=int)
args = parser.parse_args()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
args.tokenizer = tokenizer
# set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
helper.print_arguments(args)

# load vocab
print("Loading vocab...")
token_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_tok.vocab")  # token
post_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_post.vocab")  # position
pos_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_pos.vocab")  # POS
dep_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_dep.vocab")  # deprel
pol_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_pol.vocab")  # polarity
vocab = (token_vocab, post_vocab, pos_vocab, dep_vocab, pol_vocab)
print(
    "token_vocab: {}, post_vocab: {}, pos_vocab: {}, dep_vocab: {}, pol_vocab: {}".format(
        len(token_vocab), len(post_vocab), len(pos_vocab), len(dep_vocab), len(pol_vocab)
    )
)
args.tok_size = len(token_vocab)
args.post_size = len(post_vocab)
args.pos_size = len(pos_vocab)
args.dep_size = len(dep_vocab)

# load data
print("Loading data from {} with batch size {}...".format(args.data_dir, args.batch_size))
train_batch = ABSADataLoader(
    args.data_dir + "/train.json", args.batch_size, args, vocab, shuffle=True
)
valid_batch = ABSADataLoader(
    args.data_dir + "/valid.json", args.batch_size, args, vocab, shuffle=False
)
test_batch = ABSADataLoader(
    args.data_dir + "/test.json", args.batch_size, args, vocab, shuffle=False
)

# check saved_models director
model_save_dir = args.save_dir
helper.ensure_dir(model_save_dir, verbose=True)
if not os.path.exists(model_save_dir + str(args.seed)):
    os.makedirs(model_save_dir + str(args.seed))
file_logger = helper.FileLogger(model_save_dir + str(args.seed) + '/' + args.log, header="epoch\ttrain_loss\ttrain_acc")


def evaluate(model, data_loader):
    predictions, labels = [], []
    val_loss, val_acc, val_step = 0.0, 0.0, 0
    for i, batch in enumerate(data_loader):
        loss, acc, pred, label, _, _ = model.predict(batch)
        val_loss += loss
        val_acc += acc
        predictions += pred
        labels += label
        val_step += 1
    # f1 score
    f1_score = metrics.f1_score(labels, predictions, average="macro")
    return val_loss / val_step, val_acc / val_step, f1_score


def _totally_parameters(model):  #
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params



# build model
trainer = ABSATrainer(args)
# print(trainer.model)
# print('# parameters:', _totally_parameters(trainer.model))


best_path = model_save_dir

print("Training Set: {}".format(len(train_batch)))
print("Valid/Test Set: {}".format(len(test_batch)))

train_acc_history, train_loss_history = [0.0], [0.0]
val_acc_history, val_loss_history, val_f1_score_history = [0.0], [0.0], [0.0]
test_acc_history, test_loss_history, test_f1_score_history = [0.0], [0.0], [0.0]

for epoch in range(1, args.num_epoch + 1):
    print("Epoch {}".format(epoch) + "-" * 60)
    train_loss, train_acc, train_step = 0.0, 0.0, 0
    for i, batch in enumerate(train_batch):
        loss, acc = trainer.update(batch)
        train_loss += loss
        train_acc += acc
        train_step += 1
        if train_step % args.log_step == 0:
            print(
                "{}/{} train_loss: {:.6f}, train_acc: {:.6f}".format(
                    i, len(train_batch), train_loss / train_step, train_acc / train_step
                )
            )
            file_logger.log("{}\t{:.6f}\t{:.6f}\t".format(
                epoch, train_loss / train_step,train_acc / train_step))

    val_loss, val_acc, val_f1 = evaluate(trainer, valid_batch)
    writer.add_scalar("loss", train_loss / train_step, epoch)
    writer.add_scalar("accuracy", train_acc / train_step, epoch)
    writer.add_scalar("val_loss", val_loss, epoch)
    writer.add_scalar("val_acc", val_acc, epoch)

    print(
        "End of {} train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, f1_score: {:.4f}".format(
            epoch, train_loss / train_step, train_acc / train_step, val_loss, val_acc, val_f1
        )
    )
    file_logger.log("End of {} train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, f1_score: {:.4f}".format(
            epoch, train_loss / train_step, train_acc / train_step, val_loss, val_acc, val_f1
        )
    )
    train_acc_history.append(train_acc / train_step)
    train_loss_history.append(train_loss / train_step)
    val_loss_history.append(val_loss)

    # save best model
    if epoch == 1 or float(val_acc) > max(val_acc_history):
        if not os.path.exists(best_path + str(args.seed)):
            os.makedirs(best_path + str(args.seed))
        torch.save(trainer,best_path + str(args.seed) + '/best_model.pt')
        print("new best model saved.")
        file_logger.log("new best model saved at epoch {}: {:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}"
                        .format(epoch, train_loss / train_step,train_acc / train_step,val_loss,val_acc,val_f1))

    val_acc_history.append(float(val_acc))
    val_f1_score_history.append(val_f1)
    val_acc_history.append(float(val_acc))
    val_f1_score_history.append(val_f1)
    print(max(val_acc_history))

print("Training ended with {} epochs.".format(epoch))

# bt_train_acc = max(train_acc_history)
# bt_train_loss = min(train_loss_history)
# bt_val_acc = max(val_acc_history)
# bt_val_acc_idx = val_acc_history.index(bt_val_acc)
# bt_val_f1 = val_f1_score_history[bt_val_acc_idx]
# bt_val_loss = val_loss_history[bt_val_acc_idx]
# print("Training Summary: best_val_epoch:{}, valid_loss:{}, valid_acc:{}, valid_f1:{}".format(bt_val_acc_idx, bt_val_loss, bt_val_acc, bt_val_f1))
writer.close()
print("Loading best checkpoint from ", best_path + str(args.seed) + '/best_model.pt')
trainer = torch.load(best_path + str(args.seed) + '/best_model.pt')
test_loss, test_acc, test_f1 = evaluate(trainer, test_batch)
print("Evaluation Results: test_loss:{}, test_acc:{}, test_f1:{}".format(test_loss, test_acc, test_f1))
file_logger.log("Loading best checkpoint from " + best_path + str(args.seed) + '/best_model.pt')
file_logger.log("Evaluation Results: test_loss:{}, test_acc:{}, test_f1:{}".format(test_loss, test_acc, test_f1))