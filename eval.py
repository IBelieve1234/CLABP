import sys
import os
import random
import numpy as np
import argparse
import matplotlib.pyplot as plt  
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from transformers import BertModel, BertTokenizer,\
                         T5Tokenizer, T5EncoderModel,\
                         XLNetModel, XLNetTokenizer,\
                         EsmTokenizer,EsmForSequenceClassification

import numpy as np
from utils import *
from model import *

def eval(args):

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = torch.cuda.is_available()
    use_lm = args.use_lm


    if args.model == "ABP_Text_Picture_Model":
        model = ABP_Text_Picture_Model().to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    #prot_bert_bfd
    if args.lm_model == "prot_bert_bfd":
        tokenizer = BertTokenizer.from_pretrained("./Rostlab/prot_bert_bfd", do_lower_case=False, legacy=False )
        pretrained_lm = BertModel.from_pretrained("./Rostlab/prot_bert_bfd")
    #prot_bert
    if args.lm_model == "prot_bert":
        tokenizer = BertTokenizer.from_pretrained("./Rostlab/prot_bert", do_lower_case=False, legacy=False )
        pretrained_lm = BertModel.from_pretrained("./Rostlab/prot_bert")
    #prot_t5_xl_bfd
    if args.lm_model == "prot_t5_xl_bfd":
        tokenizer = T5Tokenizer.from_pretrained("./Rostlab/prot_t5_xl_bfd", do_lower_case=False, legacy=False )
        pretrained_lm = T5EncoderModel.from_pretrained("./Rostlab/prot_t5_xl_bfd")
    #prot_t5_xl_uniref50
    if args.lm_model == "prot_t5_xl_uniref50":
        tokenizer = T5Tokenizer.from_pretrained("./Rostlab/prot_t5_xl_uniref50", do_lower_case=False, legacy=False)
        pretrained_lm = T5EncoderModel.from_pretrained("./Rostlab/prot_t5_xl_uniref50")
    #prot_xlnet
    if args.lm_model == "prot_xlnet":
        tokenizer = XLNetTokenizer.from_pretrained("./Rostlab/prot_xlnet", do_lower_case=False, legacy=False )
        pretrained_lm = XLNetModel.from_pretrained("./Rostlab/prot_xlnet",mem_len=1024)
    #ProstT5
    if args.lm_model == "ProstT5":
        tokenizer = T5Tokenizer.from_pretrained("./Rostlab/ProstT5", do_lower_case=False, legacy=False )
        pretrained_lm = T5EncoderModel.from_pretrained("./Rostlab/ProstT5")
    #esm2_t6_8M_UR50D
    if args.lm_model == "esm2_t6_8M_UR50D":
        pretrained_lm = EsmForSequenceClassification.from_pretrained("./facebook/esm2_t6_8M_UR50D", num_labels=320)
        tokenizer = EsmTokenizer.from_pretrained("./facebook/esm2_t6_8M_UR50D")
    if args.lm_model == "esm2_t33_650M_UR50D":
        pretrained_lm = EsmForSequenceClassification.from_pretrained("./facebook/esm2_t33_650M_UR50D", num_labels=1024)
        tokenizer = EsmTokenizer.from_pretrained("./facebook/esm2_t33_650M_UR50D")

    model_filename = args.model_ck_filename
    model_path = os.path.join("./checkpoints/", model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load_state_dict(torch.load(model_path))  
    model.eval()  

    dir_name = args.dir_name
    eval_seq_name = dir_name+"seq.npy"
    eval_phipsi_name = dir_name+"phipsi.npy"
    eval_DSSP_name = dir_name+"DSSP.npy"
    eval_distance_name = dir_name+"distance_value.npy"
    eval_movement_name = dir_name+"movement_vector.npy"
    eval_quate_name = dir_name+"quater_number.npy"
    eval_mask_name = dir_name+"mask.npy"
    eval_label_name = dir_name+"label.npy"
    
    eval_seqs_np = np.load(eval_seq_name).astype("long")
    eval_phipsi_np = np.load(eval_phipsi_name).astype("float32")
    eval_DSSP_np = np.load(eval_DSSP_name).astype("long")
    eval_distance_np = np.load(eval_distance_name).astype("float32")
    eval_movement_np = np.load(eval_movement_name).astype("float32")
    eval_quate_np = np.load(eval_quate_name).astype("float32")
    eval_mask_np = np.load(eval_mask_name).astype("bool")
    eval_label_np = np.load(eval_label_name).astype("int32")


    eval_seqs = torch.from_numpy(eval_seqs_np)
    eval_phipsi = torch.from_numpy(eval_phipsi_np)
    eval_DSSP = torch.from_numpy(eval_DSSP_np)
    eval_distance = torch.from_numpy(eval_distance_np)
    eval_movement = torch.from_numpy(eval_movement_np)
    eval_quate = torch.from_numpy(eval_quate_np)
    eval_mask = torch.from_numpy(eval_mask_np)
    eval_label = torch.from_numpy(eval_label_np)

    eval_set = torch.utils.data.TensorDataset(eval_seqs,eval_distance,eval_label,eval_phipsi,eval_DSSP,eval_movement,eval_quate,eval_mask)
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=args.eval_batch,shuffle=True)  

    acc_test_history = []
    pr_test_history = []
    sp_test_history = []
    auc_test_history = []
    f1_test_history = []
    mcc_test_history = []
    recall_test_history = []

    for epoch in range(1, args.num_epoch + 1):
        model.eval()  
        pretrained_lm.eval()
        acc_test = 0.0
        pr_test = 0.0
        sp_test = 0.0
        auc_test = 0.0
        f1_test = 0.0
        mcc_test = 0.0
        recall_test = 0.0
        i = 0
        with torch.no_grad():
            for seqs, distance, label, phipsi, DSSP, movement, quate, mask in eval_loader:
                seqs, distance, label, phipsi, DSSP, movement, quate, mask = \
                seqs.to(device), distance.to(device), label.float().to(device), phipsi.to(device), DSSP.to(device),\
                movement.to(device), quate.to(device), mask.to(device)
                
                pretrained_lm = pretrained_lm.to(device)
                
                if args.model == "ABP_Text_Picture_Model":
                    output, contrastive_loss = model(device,seqs,args.use_GNN,args.distance_threshold, phipsi, DSSP, movement, quate, distance, mask, use_lm = args.use_lm,pretrained_lm=pretrained_lm,tokenizer=tokenizer)
                else:
                    output = model(device,seqs,args.use_GNN,args.distance_threshold, phipsi, DSSP, movement, quate, distance, mask, use_lm = args.use_lm,pretrained_lm=pretrained_lm,tokenizer=tokenizer)
                predict = output

                acc_test += accuracy(predict=predict, label=label)
                pr_test += precision(predict=predict, label=label)
                sp_test += specificity(predict=predict, label=label)
                auc_test += auc(predict=predict, label=label)
                f1_test += f1_score(predict=predict, label=label)
                mcc_test += mcc(predict=predict, label=label)
                recall_test += recall(predict=predict, label=label)
                i += 1
        
        acc_test = acc_test / i
        pr_test = pr_test / i
        sp_test = sp_test / i
        auc_test = auc_test / i
        f1_test = f1_test / i
        mcc_test = mcc_test / i
        recall_test = recall_test / i
        acc_test_history.append(acc_test)
        pr_test_history.append(pr_test)
        sp_test_history.append(sp_test)
        auc_test_history.append(auc_test)
        f1_test_history.append(f1_test)
        mcc_test_history.append(mcc_test)
        recall_test_history.append(recall_test)
        print("epoch %-2.0f  eval_acc %-8.5f eval_pr %-8.5f eval_sp %-8.5f eval_auc %-8.5f eval_f1 %-8.5f eval_mcc %-8.5f eval_recall %-8.5f" % 
      (epoch, acc_test, pr_test, sp_test, auc_test, f1_test, mcc_test, recall_test))

    avg_acc_test = np.mean(acc_test_history)
    print(f"Average Eval accuracy: {avg_acc_test:.4f}")
    avg_pr_test = np.mean(pr_test_history)
    print(f"Average Eval precision: {avg_pr_test:.4f}")
    avg_sp_test = np.mean(sp_test_history)
    print(f"Average Eval specificity: {avg_sp_test:.4f}")
    avg_auc_test = np.mean(auc_test_history)
    print(f"Average Eval auc: {avg_auc_test:.4f}")
    avg_f1_test = np.mean(f1_test_history)
    print(f"Average Eval f1-score: {avg_f1_test:.4f}")
    avg_mcc_test = np.mean(mcc_test_history)
    print(f"Average Eval Mcc: {avg_mcc_test:.4f}")
    avg_recall_test = np.mean(recall_test_history)
    print(f"Average Eval Recall: {avg_recall_test:.4f}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Evaluation of ABPS Model")
    argparser.add_argument('--seed', default=7, type=int)#7

    argparser.add_argument("--model_ck_filename", type=str,default=f"epoch_73_use_lm_True.pt")
    argparser.add_argument('--model', default="ABP_Text_Picture_Model", type=str)
    # ABP_Text_Picture_Model

    #distance_threshold 
    argparser.add_argument("--distance_threshold", type=float,default=10)
    #if use GNN 
    argparser.add_argument("--use_GNN",type=bool,default=True)

    #lm
    argparser.add_argument("--use_lm",type=bool,default=True)
    argparser.add_argument('--lm_model', default="prot_t5_xl_uniref50", type=str)
    #prot_bert_bfd    prot_bert    prot_t5_xl_bfd    prot_t5_xl_uniref50    prot_xlnet    ProstT5    esm2_t6_8M_UR50D   esm2_t33_650M_UR50D  

    argparser.add_argument("--num_epoch", type=int, default=10, help="Number of epochs for evaluation.")
    argparser.add_argument("--eval_batch", type=int, default=256, help="Batch size for evaluation.")
    argparser.add_argument("--dir_name",type=str,default="./data/ABPDB/ABPDB_3")

    args = argparser.parse_args()
    eval(args)
