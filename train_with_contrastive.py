from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
from transformers import BertModel, BertTokenizer,\
                         T5Tokenizer, T5EncoderModel,\
                         XLNetModel, XLNetTokenizer,\
                         EsmTokenizer,EsmForSequenceClassification
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import matplotlib.pyplot as plt
import warnings
#from tqdm.auto import tqdm
import dgl
from utils import *
from model import *
import warnings
import time

warnings.filterwarnings("ignore", category=UserWarning, message="No negative samples in targets*")
warnings.filterwarnings("ignore", category=UserWarning, message="No positive samples in targets*")

def load_eval_data(dir_name="",eval_batch_size=256):
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
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=eval_batch_size,shuffle=True)  
    
    return eval_loader

def evaluate(model,eval_loader, device,pretrained_lm,tokenizer,use_lm=False):
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
    total_contrastive_loss = 0.0
    
    with torch.no_grad():
        for seqs, distance, label, phipsi, DSSP, movement, quate, mask in eval_loader:
            seqs, distance, label, phipsi, DSSP, movement, quate, mask = \
                seqs.to(device), distance.to(device), label.to(device), phipsi.to(device), DSSP.to(device), \
                movement.to(device), quate.to(device), mask.to(device)
            output, contrastive_loss = model(device,seqs,args.use_GNN,args.distance_threshold, phipsi, DSSP, movement, quate, distance, mask, use_lm = args.use_lm,pretrained_lm=pretrained_lm,tokenizer=tokenizer)
            predict = output

            # debug
            if torch.isnan(predict).any() or torch.isinf(predict).any():
                print(f"NaN : {torch.isnan(predict).sum().item()}")
                print(f"Inf : {torch.isinf(predict).sum().item()}")

            # clamp [0, 1] 
            predict = torch.clamp(predict, 0.0, 1.0)

            label = label.float().squeeze(-1)  # Squeeze to match prediction shape

            acc_test += accuracy(predict=predict, label=label)
            pr_test += precision(predict=predict, label=label)
            sp_test += specificity(predict=predict, label=label)
            auc_test += auc(predict=predict, label=label)
            f1_test += f1_score(predict=predict, label=label)
            mcc_test += mcc(predict=predict, label=label)
            recall_test += recall(predict=predict, label=label)
            total_contrastive_loss += contrastive_loss.item()
            i += 1

    acc_test = acc_test / i
    pr_test = pr_test / i
    sp_test = sp_test / i
    auc_test = auc_test / i
    f1_test = f1_test / i
    mcc_test = mcc_test / i
    recall_test = recall_test / i
    avg_contrastive_loss = total_contrastive_loss / i

    return acc_test, pr_test, sp_test, auc_test, f1_test, mcc_test, recall_test, avg_contrastive_loss

def main(args):

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    
    # ABP_Text_Picture_Model
    if args.model == "ABP_Text_Picture_Model":
        model = ABP_Text_Picture_Model().to(device)
        args.use_GNN = True
        args.train_batch = 128 #128
        args.distance_threshold = 10 
        args.lr = 0.0001875/4
        args.eval_batch = 2 * args.train_batch
        args.independent_eval_batch = 2 * args.train_batch 
        args.eval_1_interval_strat_epoch = 1
        args.num_epoch = 120#45 50
    
    if (args.use_seq == False):
        args.use_lm = False
    print("---------start------------")
    print("device: "+str(device))
    print("Seed: " + str(args.seed))
    print("Model: "+args.model)
    print("use_seq: "+str(args.use_seq))
    if (args.model == "ABP_Text_Picture_Model"):
        print("Contrastive Learning Weight: "+str(args.contrastive_weight))
    
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

    if args.use_lm is True:
        print("lm_Model: "+str(args.lm_model))
    else:
        print("lm_Model: onehot")

    pretrained_lm = pretrained_lm.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=0.0005)
    pretrain_inner_opt = torch.optim.SGD(pretrained_lm.parameters(),lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    pretrain_scheduler = StepLR(pretrain_inner_opt, step_size=args.step_size, gamma=args.gamma)  # 预训练模型的调度器

    start_epoch = 1  # start epoch
    if args.checkpoint is not None:
        print(f"Loading checkpoint from: {args.checkpoint}")
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Model weights loaded successfully")

                if args.load_optimizer and 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("Optimizer state loaded successfully")

                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("Scheduler state loaded successfully")

                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                    print(f"Resuming from epoch {start_epoch}")

            else:
                model.load_state_dict(checkpoint)
                print("Model weights loaded successfully")

            print(f"Checkpoint loaded. Starting from epoch {start_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Training from scratch...")
    # ========================================

    loss_func =  torch.nn.BCELoss()#
    #loss_func = torch.nn.BCEWithLogitsLoss()
    use_gpu = device.type == 'cuda'

    loss_history = []
    contrastive_loss_history = []  # 
    acc_test_history = []

    #Train_loader, Test_loader = loaddataset()
    subset = "train"
    dir_name = args.dir_name+"ABPDB_7/"
    
    seq_name = dir_name+"seq.npy"                                                       #protein.py
    phipsi_name = dir_name+"phipsi.npy"                                                 #protein.py
    DSSP_name = dir_name+"DSSP.npy"                                                     #protein.py
    distance_name = dir_name+"distance_value.npy"                                       #protein.py
    movement_name = dir_name+"movement_vector.npy"                                      #protein.py
    quate_name = dir_name+"quater_number.npy"                                           #protein.py
    mask_name = dir_name+"mask.npy"                                                     #protein.py
    label_name = dir_name+"label.npy"                                                   #protein.py

    seqs_np = np.load(seq_name).astype("long")                                          #protein.py  
    phipsi_np = np.load(phipsi_name).astype("float32")                                  #protein.py
    DSSP_np = np.load(DSSP_name).astype("long")                                         #protein.py
    distance_np = np.load(distance_name).astype("float32")                              #protein.py
    movement_np = np.load(movement_name).astype("float32")                              #protein.py
    quate_np = np.load(quate_name).astype("float32")                                    #protein.py
    mask_np = np.load(mask_name).astype("bool")                                         #protein.py
    label_np = np.load(label_name).astype("int32")                                      #protein.py

    seqs = torch.from_numpy(seqs_np)
    phipsi = torch.from_numpy(phipsi_np)
    DSSP = torch.from_numpy(DSSP_np)
    distance = torch.from_numpy(distance_np)
    movement = torch.from_numpy(movement_np)
    quate = torch.from_numpy(quate_np)
    mask = torch.from_numpy(mask_np)
    label = torch.from_numpy(label_np)

    del seqs_np,phipsi_np,DSSP_np,distance_np,movement_np,quate_np,mask_np,label_np

    # node feature
    if (args.use_seq == False):
        seqs = torch.zeros_like(seqs)
        args.use_lm = False
    if (args.use_phipsi == False):
        phipsi = torch.zeros_like(phipsi)
    if (args.use_DSSP == False):
        DSSP = torch.zeros_like(DSSP)
        #edge feature
    if(args.use_movement == False):
        movement = torch.zeros_like(movement)
    if(args.use_quate == False):
        quate = torch.zeros_like(quate)
    if(args.use_distance == False):
        distance = torch.zeros_like(distance)
    
    train_num = int(args.data_num * 0.9)
    print(f"train_num:{train_num}")
    test_num =args.data_num-int(args.data_num * 0.9)
    print(f"test_num:{test_num}")
    Total_set = torch.utils.data.TensorDataset(seqs,distance,label,phipsi,DSSP,movement,quate,mask)
    Train_set , Test_set = torch.utils.data.random_split(Total_set,[train_num,test_num])
    Train_loader = torch.utils.data.DataLoader(Train_set,batch_size=args.train_batch)#batch_size
    Test_loader = torch.utils.data.DataLoader(Test_set,batch_size=args.test_batch)#batch_size
    print("train batch_size: "+str(args.train_batch))
    print("train lr: "+str(args.lr))
    print("step_size: "+str(args.step_size))
    print("gamma: "+str(args.gamma))
    print("eval batch_size: "+str(args.eval_batch))
    print("indepedent batch_size: "+str(args.independent_eval_batch))
    print("finish loading train dataset")

    num_epoch = args.num_epoch
    best_acc = 0.0
    best_epoch = 0
    best_model = model

    for epoch in range(start_epoch, num_epoch+1):
        model.train()
        pretrained_lm.train()
        if(use_gpu):
            model.cuda()
            pretrained_lm.cuda()
        step = 0
        epoch_classification_loss = 0.0
        epoch_contrastive_loss = 0.0
        
        for seqs,distance,label,phipsi,DSSP,movement,quate,mask in Train_loader:
            step += 1
            if(use_gpu):
                seqs, distance, label, phipsi, DSSP, movement, quate, mask = \
                    seqs.cuda(), distance.cuda(), label.cuda(), phipsi.cuda(), DSSP.cuda(), \
                    movement.cuda(), quate.cuda(), mask.cuda()
                
            optimizer.zero_grad()
            if args.use_lm is True:
                pretrain_inner_opt.zero_grad()
            

            if (args.model == "ABP_Text_Picture_Model"):
                output, contrastive_loss = model(device,seqs,args.use_GNN,args.distance_threshold, phipsi, DSSP, movement, quate, distance, mask, use_lm = args.use_lm,pretrained_lm=pretrained_lm,tokenizer=tokenizer)
            else:
                output = model(device,seqs,args.use_GNN,args.distance_threshold, phipsi, DSSP, movement, quate, distance, mask, use_lm = args.use_lm,pretrained_lm=pretrained_lm,tokenizer=tokenizer)
                contrastive_loss = torch.tensor(0.0).to(device)  

            predict = output
            
            """
            print(f"predict shape: {predict.shape}")
            print(f"label shape: {label.shape}")
            print(f"predict min: {predict.min().item():.6f}, max: {predict.max().item():.6f}")
            """
            # predict [0,1]
            predict = torch.clamp(predict, 0.0, 1.0)
            label = label.float().squeeze(-1)  # Squeeze to match prediction shape [batch_size]
            acc = accuracy(predict=predict,label=label)
            classification_loss = loss_func(predict, label)
            total_loss = classification_loss + args.contrastive_weight * contrastive_loss
            loss = total_loss 
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if args.use_lm is True:
                pretrain_inner_opt.step()
    
            epoch_classification_loss += classification_loss.item()
            epoch_contrastive_loss += contrastive_loss.item()
            
        if epoch >= args.eval_1_interval_strat_epoch:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, f"./checkpoints/epoch_{epoch}_use_lm_{args.use_lm}_full.pt")
            torch.save(model.state_dict(), f"./checkpoints/epoch_{epoch}_use_lm_{args.use_lm}.pt")
        eval_interval = args.eval_interval
        if epoch % eval_interval == 0:            
            #"""
            ABPDB_dir = args.dir_name+"ABPDB_3/"               
            ABPDB_loader = load_eval_data(ABPDB_dir, int(args.independent_eval_batch))  
            acc_ABPDB, pr_ABPDB, sp_ABPDB, auc_ABPDB, f1_ABPDB, mcc_ABPDB, recall_ABPDB, contrastive_ABPDB = evaluate(model, ABPDB_loader, device, pretrained_lm,tokenizer,use_lm = args.use_lm)
            print("ABPDB_epoch %-2.0f ABPDB_acc %-8.5f ABPDB_pr %-8.5f ABPDB_sp %-8.5f ABPDB_auc %-8.5f ABPDB_f1 %-8.5f ABPDB_mcc %-8.5f ABPDB_recall %-8.5f ABPDB_contrastive %-8.5f" % 
      (epoch, acc_ABPDB, pr_ABPDB, sp_ABPDB, auc_ABPDB, f1_ABPDB, mcc_ABPDB, recall_ABPDB, contrastive_ABPDB))
            model.train()
            #"""

        scheduler.step()
        if args.use_lm is True:
            pretrain_scheduler.step()
        acc_test = 0.0
        pr_test = 0.0
        sp_test = 0.0
        auc_test = 0.0
        f1_test = 0.0
        mcc_test = 0.0
        recall_test = 0.0
        test_contrastive_loss = 0.0
        i = 0
        with torch.no_grad():
            for seqs,distance,label,phipsi,DSSP,movement,quate,mask in Test_loader:
                step += 1
                if(use_gpu):
                    seqs, distance, label, phipsi, DSSP, movement, quate, mask = \
                        seqs.cuda(), distance.cuda(), label.cuda(), phipsi.cuda(), DSSP.cuda(), \
                        movement.cuda(), quate.cuda(), mask.cuda()
                
                if (args.model == "ABP_Text_Picture_Model"):
                    output, contrastive_loss = model(device,seqs,args.use_GNN,args.distance_threshold, phipsi, DSSP, movement, quate, distance, mask, use_lm = args.use_lm,pretrained_lm=pretrained_lm,tokenizer=tokenizer)
                    test_contrastive_loss += contrastive_loss.item()
                else:
                    output = model(device,seqs,args.use_GNN,args.distance_threshold, phipsi, DSSP, movement, quate, distance, mask, use_lm = args.use_lm,pretrained_lm=pretrained_lm,tokenizer=tokenizer)

                predict = output
                label = label.float().squeeze(-1)  # Squeeze to match prediction shape
                acc_test += accuracy(predict=predict,label=label)
                pr_test += precision(predict=predict,label=label)
                sp_test += specificity(predict=predict,label=label)
                auc_test += auc(predict=predict,label=label)
                f1_test += f1_score(predict=predict,label=label)
                mcc_test += mcc(predict=predict,label=label)
                recall_test += recall(predict=predict,label=label)
                i += 1
            acc_test = acc_test / i
            pr_test = pr_test/i
            sp_test = sp_test/i
            auc_test = auc_test/i
            f1_test = f1_test/i
            mcc_test = mcc_test/i
            recall_test = recall_test/i
            test_contrastive_loss = test_contrastive_loss / i

        avg_classification_loss = epoch_classification_loss / len(Train_loader)
        avg_contrastive_loss = epoch_contrastive_loss / len(Train_loader)
        loss_history.append(avg_classification_loss)
        contrastive_loss_history.append(avg_contrastive_loss)
        acc_test_history.append(acc_test)

        if (args.model == "ABP_Text_Picture_Model"):
            print("epoch %-2.0f cls_loss %-8.5f cont_loss %-8.5f total_loss %-8.5f Train_acc %-8.5f Test_acc %-8.5f Test_pr %-8.5f Test_sp %-8.5f Test_auc %-8.5f Test_f1 %-8.5f Test_mcc %-8.5f Test_recall %-8.5f Test_cont_loss %-8.5f" %
          (epoch, avg_classification_loss, avg_contrastive_loss, avg_classification_loss + args.contrastive_weight * avg_contrastive_loss, acc, acc_test, pr_test, sp_test, auc_test, f1_test, mcc_test, recall_test, test_contrastive_loss))
        else:
            print("epoch %-2.0f loss %-8.5f Train_acc %-8.5f Test_acc %-8.5f Test_pr %-8.5f Test_sp %-8.5f Test_auc %-8.5f Test_f1 %-8.5f Test_mcc %-8.5f Test_recall %-8.5f" %
          (epoch, avg_classification_loss, acc, acc_test, pr_test, sp_test, auc_test, f1_test, mcc_test, recall_test))
        
        if((acc_test>best_acc)&(epoch>=0.9*args.num_epoch)):
            best_model = model
            best_epoch = epoch
            best_acc = acc_test
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }
            torch.save(best_checkpoint, f"./checkpoints/tmp_use_lm_{args.use_lm}_full.pt")
            torch.save(best_model.state_dict(), f"./checkpoints/tmp_use_lm_{args.use_lm}.pt")
    print("best epoch:",end=" ")
    print(best_epoch)
    print("acc:",end=" ")
    print(best_acc)
    final_best_checkpoint = {
        'epoch': best_epoch,
        'model_state_dict': best_model.state_dict(),
        'best_acc': best_acc,
    }
    torch.save(final_best_checkpoint, f"./checkpoints/{args.use_lm}_contrastive_full.pt")
    torch.save(best_model.state_dict(), f"./checkpoints/{args.use_lm}_contrastive.pt")


    del seqs,phipsi,DSSP,distance,movement,quate,mask,label
    print("---------end------------")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="ABPModel with Contrastive Learning")
    argparser.add_argument('--seed', default=7, type=int)#7
    argparser.add_argument('--model', default="ABP_Text_Picture_Model", type=str)   
    argparser.add_argument('--checkpoint', default=None, type=str, help="Path to checkpoint file to resume training from") # ./checkpoints/epoch_73_use_lm_True.pt
    argparser.add_argument('--load_optimizer', default=False, type=bool, help="Whether to load optimizer state from checkpoint")

    argparser.add_argument('--contrastive_weight', default=0.1, type=float, help="Weight for contrastive learning loss")
    
    #node feature
    argparser.add_argument("--use_seq", type=bool,default=True, help="sequence.")
    argparser.add_argument("--use_phipsi", type=bool,default=True, help="phipsi.")
    argparser.add_argument("--use_DSSP", type=bool,default=True, help="DSSP.")
    #edge feature
    argparser.add_argument("--use_distance", type=bool,default=True, help="distance.")
    argparser.add_argument("--use_movement", type=bool,default=True, help="movement.")
    argparser.add_argument("--use_quate", type=bool,default=True, help="quate.")
    #lm
    argparser.add_argument("--use_lm",type=bool,default=True)
    argparser.add_argument('--lm_model', default="prot_t5_xl_uniref50", type=str)
    #prot_bert_bfd    prot_bert    prot_t5_xl_bfd    prot_t5_xl_uniref50    prot_xlnet    ProstT5    esm2_t6_8M_UR50D   esm2_t33_650M_UR50D  

    #distance_thresholdFalse
    argparser.add_argument("--distance_threshold", type=float,default=10)#
    #if use GNN
    argparser.add_argument("--use_GNN",type=bool,default=False)
    
    argparser.add_argument("--num_epoch", type=int,default=100)#
    argparser.add_argument("--lr", type=float,default=0.0001875/2)#0.0001875/4  0.000004
    argparser.add_argument("--step_size",type=int,default=47)
    argparser.add_argument("--gamma", type=float,default=0.618)
    argparser.add_argument("--eval_batch",type=int,default=256)
    argparser.add_argument("--independent_eval_batch",type=int,default=128)
    argparser.add_argument("--train_batch",type=int,default=128)#128
    argparser.add_argument("--test_batch",type=int,default=16)
    argparser.add_argument("--eval_interval",type=int,default=1)#eval 
    argparser.add_argument("--eval_1_interval_strat_epoch",type=int,default=1)
    argparser.add_argument("--data_num",type=int,default=4471)
    argparser.add_argument("--dir_name",type=str,default="./data/ABPDB/")  

    args = argparser.parse_args()  
    main(args)
