#!/bin/bash

# model_dir=model_myco_nucpol_pt2
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch BassenjiMultiNetwork -mid -v
# model_dir=model_myco_nucpol_pt3
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch BassenjiMultiNetwork -mid -v
# model_dir=model_myco_nucpol_pt4
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch BassenjiMultiNetwork -mid -v

# model_dir=model_myco_nucpol_pt5
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch BassenjiMultiNetwork \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -b 8192 -mt 32 -mv 8
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch BassenjiMultiNetwork -mid -v
# model_dir=model_myco_nucpol_pt6
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch BassenjiMultiNetwork \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -b 4096 -mt 64 -mv 16
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch BassenjiMultiNetwork -mid -v
# model_dir=model_myco_nucpol_pt7
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch BassenjiMultiNetwork \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -b 2048 -mt 128 -mv 32
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch BassenjiMultiNetwork -mid -v

# model_dir=model_myco_pol_pt1
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch BassenjiMultiNetwork \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch BassenjiMultiNetwork -mid -v
# model_dir=model_myco_nuc_pt5
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw \
#     -o $model_dir -arch BassenjiMultiNetwork \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch BassenjiMultiNetwork -mid -v

# model_dir=model_myco_nucpol_pt8
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch BassenjiMultiNetwork2 \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch BassenjiMultiNetwork2 -mid -v -nt 2
# model_dir=model_myco_pol_pt8
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch BassenjiMultiNetwork2 \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch BassenjiMultiNetwork2 -mid -v
# model_dir=model_myco_nuc_pt8
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw \
#     -o $model_dir -arch BassenjiMultiNetwork2 \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch BassenjiMultiNetwork2 -mid -v

# model_dir=model_myco_nucpol_pt9
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch OriginalBassenjiMultiNetwork -crop 8 \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch OriginalBassenjiMultiNetwork -crop 8 -mid -v -nt 2
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch OriginalBassenjiMultiNetwork -crop 8 -v -nt 2
# model_dir=model_myco_pol_pt9
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch OriginalBassenjiMultiNetwork -crop 8 \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch OriginalBassenjiMultiNetwork -crop 8 -mid -v
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch OriginalBassenjiMultiNetwork -crop 8 -v

# model_dir=model_myco_nuc_pt9
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw \
#     -o $model_dir -arch OriginalBassenjiMultiNetwork -crop 8 \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch OriginalBassenjiMultiNetwork -crop 8 -mid -v
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch OriginalBassenjiMultiNetwork -crop 8 -v
# model_dir=model_myco_nuc_pt10
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw \
#     -o $model_dir -arch OriginalBassenjiMultiNetwork2 -crop 8 \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch OriginalBassenjiMultiNetwork2 -crop 8 -mid -v
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch OriginalBassenjiMultiNetwork2 -crop 8 -v
# model_dir=model_myco_nuc_pt11
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw \
#     -o $model_dir -arch OriginalBassenjiMultiNetworkNoCrop \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch OriginalBassenjiMultiNetworkNoCrop -mid -v
# model_dir=model_myco_nuc_pt12
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw \
#     -o $model_dir -arch BassenjiMultiNetworkCrop -crop 8 \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch BassenjiMultiNetworkCrop -crop 8 -mid -v
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch BassenjiMultiNetworkCrop -crop 8 -v
# model_dir=model_myco_nuc_pt13
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw \
#     -o $model_dir -arch BassenjiMultiNetwork \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch BassenjiMultiNetwork -mid -v

# arch=BassenjiMultiNetwork
# model_dir=model_myco_nucpol_pt14
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch $arch \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 4096
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -mid -v -nt 2 -w 4096
# model_dir=model_myco_pol_pt14
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch $arch \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 4096
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -mid -v -w 4096
# model_dir=model_myco_nuc_pt14
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw \
#     -o $model_dir -arch $arch \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 4096
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -mid -v -w 4096

# model_dir=model_myco_nucpol_pt15
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch $arch \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 8192
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -mid -v -nt 2 -w 8192
# model_dir=model_myco_pol_pt15
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch $arch \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 8192
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -mid -v -w 8192
# model_dir=model_myco_nuc_pt15
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw \
#     -o $model_dir -arch $arch \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 8192
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -mid -v -w 8192

# model_dir=model_myco_nucpol_pt16
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch $arch \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 16384
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -mid -v -nt 2 -w 16384
# model_dir=model_myco_pol_pt16
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch $arch \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 16384
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -mid -v -w 16384
# model_dir=model_myco_nuc_pt16
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw \
#     -o $model_dir -arch $arch \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 16384
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -mid -v -w 16384

# arch=BassenjiMultiNetwork2
# model_dir=model_myco_nucpol_pt17
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch $arch \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 4096
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -mid -v -nt 2 -w 4096 -b 2048
# model_dir=model_myco_pol_pt17
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch $arch \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 4096
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -mid -v -w 4096 -b 2048
# model_dir=model_myco_nuc_pt17
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw \
#     -o $model_dir -arch $arch \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 4096
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -mid -v -w 4096 -b 2048

# model_dir=model_myco_nucpol_pt18
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch $arch \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 8192
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -mid -v -nt 2 -w 8192 -b 2048
# model_dir=model_myco_pol_pt18
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch $arch \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 8192
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -mid -v -w 8192 -b 2048
# model_dir=model_myco_nuc_pt18
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw \
#     -o $model_dir -arch $arch \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 8192
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -mid -v -w 8192 -b 2048

# model_dir=model_myco_nucpol_pt19
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch $arch \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 16384
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -mid -v -nt 2 -w 16384 -b 2048
# model_dir=model_myco_pol_pt19
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch $arch \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 16384
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -mid -v -w 16384 -b 2048
# model_dir=model_myco_nuc_pt19
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw \
#     -o $model_dir -arch $arch \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 16384
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -mid -v -w 16384 -b 2048

# arch=OriginalBassenjiMultiNetwork
# model_dir=model_myco_nucpol_pt20
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch $arch -crop 8 \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 4096
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -crop 8 -mid -v -nt 2 -w 4096
# model_dir=model_myco_pol_pt20
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch $arch -crop 8 \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 4096
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -crop 8 -mid -v -w 4096
# model_dir=model_myco_nuc_pt20
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw \
#     -o $model_dir -arch $arch -crop 8 \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 4096
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -crop 8 -mid -v -w 4096

# model_dir=model_myco_nucpol_pt21
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch $arch -crop 8 \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 8192
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -crop 8 -mid -v -nt 2 -w 8192
# model_dir=model_myco_pol_pt21
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch $arch -crop 8 \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 8192
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -crop 8 -mid -v -w 8192
# model_dir=model_myco_nuc_pt21
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw \
#     -o $model_dir -arch $arch -crop 8 \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 8192
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -crop 8 -mid -v -w 8192

# model_dir=model_myco_nucpol_pt22
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch $arch -crop 8 \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 16384
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -crop 8 -mid -v -nt 2 -w 16384
# model_dir=model_myco_pol_pt22
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o $model_dir -arch $arch -crop 8 \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 16384
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -crop 8 -mid -v -w 16384
# model_dir=model_myco_nuc_pt22
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw \
#     -o $model_dir -arch $arch -crop 8 \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 16384
# python predict_pytorch.py -m $model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o $model_dir -arch $arch -crop 8 -mid -v -w 16384

# python kMC_sequence_design_pytorch.py \
#     -o generated/4kb_regnuc_1seq_randomflanks_b16384 \
#     -m model_myco_nuc_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 16384 \
#     -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     -n 1 -l 4000 --steps 500 -t 0.0001 -s 16 --flanks random -ilen 0 -per 167 -plen 147 -pshape gaussian --seed 0 -v
# python kMC_sequence_design_pytorch.py \
#     -o generated/4kb_regnuc_1seq_randomflanks_b1024 \
#     -m model_myco_nuc_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     -n 1 -l 4000 --steps 500 -t 0.0001 -s 16 --flanks random -ilen 0 -per 167 -plen 147 -pshape gaussian --seed 0 -v
# python kMC_sequence_design_pytorch.py \
#     -o generated/4kb_regnuc_10seq_randomflanks_b16384 \
#     -m model_myco_nuc_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 16384 \
#     -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     -n 10 -l 4000 --steps 500 -t 0.0001 -s 16 --flanks random -ilen 0 -per 167 -plen 147 -pshape gaussian --seed 0 -v
# python kMC_sequence_design_pytorch.py \
#     -o generated/4kb_regnuc_10seq_randomflanks_b1024 \
#     -m model_myco_nuc_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     -n 10 -l 4000 --steps 500 -t 0.0001 -s 16 --flanks random -ilen 0 -per 167 -plen 147 -pshape gaussian --seed 0 -v
# python kMC_sequence_design_pytorch.py \
#     -o generated/20kb_regnuc_1seq_randomflanks_b16384 \
#     -m model_myco_nuc_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 16384 \
#     -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     -n 1 -l 20000 --steps 500 -t 0.0001 -s 16 --flanks random -ilen 0 -per 167 -plen 147 -pshape gaussian --seed 0 -v
# python kMC_sequence_design_pytorch.py \
#     -o generated/20kb_regnuc_1seq_randomflanks_b1024 \
#     -m model_myco_nuc_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     -n 1 -l 20000 --steps 500 -t 0.0001 -s 16 --flanks random -ilen 0 -per 167 -plen 147 -pshape gaussian --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o generated/4kb_regnuc_1seq_randomflanks_model_myco_nuc_pt17 \
#     -m model_myco_nuc_pt17/model_state.pt \
#     -w 4096 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     -n 1 -l 4000 --steps 500 -t 0.0001 -s 16 --flanks random -ilen 0 -per 167 -plen 147 -pshape gaussian --seed 0 -v
# python kMC_sequence_design_pytorch.py \
#     -o generated/4kb_regnuc_10seq_randomflanks_b16384_nw8 \
#     -m model_myco_nuc_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 16384 -nw 8 \
#     -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     -n 10 -l 4000 --steps 500 -t 0.0001 -s 16 --flanks random -ilen 0 -per 167 -plen 147 -pshape gaussian --seed 0 -v
# python kMC_sequence_design_pytorch.py \
#     -o generated/4kb_regnuc_10seq_randomflanks_b1024_nw8 \
#     -m model_myco_nuc_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 -nw 8 \
#     -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     -n 10 -l 4000 --steps 500 -t 0.0001 -s 16 --flanks random -ilen 0 -per 167 -plen 147 -pshape gaussian --seed 0 -v
# python kMC_sequence_design_pytorch.py \
#     -o generated/4kb_regnuc_10seq_randomflanks_model_myco_nuc_pt17 \
#     -m model_myco_nuc_pt17/model_state.pt \
#     -w 4096 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 8192 -nw 8 \
#     -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     -n 10 -l 4000 --steps 500 -t 0.0001 -s 16 --flanks random -ilen 0 -per 167 -plen 147 -pshape gaussian --seed 0 -v
# python kMC_sequence_design_pytorch.py \
#     -o generated/20kb_regnuc_1seq_randomflanks_model_myco_nuc_pt17 \
#     -m model_myco_nuc_pt17/model_state.pt \
#     -w 4096 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 8192 -nw 8 \
#     -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     -n 1 -l 20000 --steps 500 -t 0.0001 -s 16 --flanks random -ilen 0 -per 167 -plen 147 -pshape gaussian --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o generated/regnuc_2kb_1seq_randomflanks \
#     -m Trainedmodels/model_myco_nuc_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     -n 1 -l 2000 --steps 500 -t 0.0001 -s 16 --flanks random -ilen 0 -per 167 -plen 147 -pshape gaussian --seed 0 -v
# python kMC_sequence_design_pytorch.py \
#     -o generated/regnuc_2kb_10seq_randomflanks \
#     -m Trainedmodels/model_myco_nuc_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     --flanks random \
#     -n 10 -l 2000 --steps 500 -t 0.0001 -s 16 -ilen 0 -per 167 -plen 147 -pshape gaussian --seed 0 -v
# python kMC_sequence_design_pytorch.py \
#     -o generated/regnuc_2kb_10seq_flanksInt2 \
#     -m Trainedmodels/model_myco_nuc_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     --flanks /home/alex/shared_folder/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     -n 10 -l 2000 --steps 500 -t 0.0001 -s 16 -ilen 0 -per 167 -plen 147 -pshape gaussian --seed 0 -v
# python kMC_sequence_design_pytorch.py \
#     -o generated/regnuc_2kb_100seq_randomflanks \
#     -m Trainedmodels/model_myco_nuc_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     --flanks random \
#     -n 100 -l 2000 --steps 500 -t 0.0001 -s 16 -ilen 0 -per 167 -plen 147 -pshape gaussian --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/nucNDR_simpleNDR_2kb_100seq_flanksInt2 \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nuc_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     --start_seqs /home/alex/SCerevisiae_chromatin_NN_prediction/generated/regnuc_2kb_100seq_randomflanks/100_selected_seqs.npy \
#     --flanks /home/alex/shared_folder/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     --target_file /home/alex/SCerevisiae_chromatin_NN_prediction/generated/regnuc_2kb_100seq_randomflanks/target_with_NDR.npz \
#     --steps 500 -t 0.0001 -s 16 --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/polsigmoid_right_fromregnuc_2kb_100seq_flanksInt2 \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nucpol_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 -nt 2 -track 1 \
#     --start_seqs /home/alex/SCerevisiae_chromatin_NN_prediction/generated/regnuc_2kb_100seq_randomflanks/100_selected_seqs.npy \
#     --flanks /home/alex/shared_folder/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     --steps 500 -t 0.0001 -s 16 -ilen 500 -ishape sigmoid -bg low high --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/polpeak_200bp_fromregnuc_2kb_100seq_flanksInt2 \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nucpol_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 -nt 2 -track 1 \
#     --start_seqs /home/alex/SCerevisiae_chromatin_NN_prediction/generated/regnuc_2kb_100seq_randomflanks/100_selected_seqs.npy \
#     --flanks /home/alex/shared_folder/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     --steps 500 -t 0.0001 -s 16 -ilen 200 -ishape gaussian --seed 0 -v
# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/polpeak_500bp_fromregnuc_2kb_100seq_flanksInt2 \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nucpol_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 -nt 2 -track 1 \
#     --start_seqs /home/alex/SCerevisiae_chromatin_NN_prediction/generated/regnuc_2kb_100seq_randomflanks/100_selected_seqs.npy \
#     --flanks /home/alex/shared_folder/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     --steps 500 -t 0.0001 -s 16 -ilen 500 -ishape gaussian --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/nucNDR_simpleNDR_newphasing_2kb_100seq_flanksInt2 \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nuc_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     --start_seqs /home/alex/SCerevisiae_chromatin_NN_prediction/generated/regnuc_2kb_100seq_randomflanks/100_selected_seqs.npy \
#     --flanks /home/alex/shared_folder/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     -ilen 147 -per 167 -plen 147 -pshape gaussian \
#     --steps 500 -t 0.0001 -s 16 --seed 0 -v
    
# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/nucNDR_doubleNDR_2kb_100seq_flanksInt2 \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nuc_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     --start_seqs /home/alex/SCerevisiae_chromatin_NN_prediction/generated/regnuc_2kb_100seq_randomflanks/100_selected_seqs.npy \
#     --flanks /home/alex/shared_folder/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     --target_file /home/alex/SCerevisiae_chromatin_NN_prediction/generated/regnuc_2kb_100seq_randomflanks/target_with_NDR2.npz \
#     --steps 500 -t 0.0001 -s 16 --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o generated/regnuc_2kb_100seq_randomflanks_gclen100 \
#     -m Trainedmodels/model_myco_nuc_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     --flanks random \
#     -gctol 0 -gclen 100 \
#     -n 100 -l 2000 --steps 500 -t 0.0001 -s 16 -ilen 0 -per 167 -plen 147 -pshape gaussian --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o generated/regnuc_2kb_100seq_flanksInt2_gclen100 \
#     -m Trainedmodels/model_myco_nuc_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     --flanks /home/alex/shared_folder/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     -gctol 0.01 -gclen 100 \
#     -n 100 -l 2000 --steps 500 -t 0.0001 -s 16 -ilen 0 -per 167 -plen 147 -pshape gaussian --seed 1 -v

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/nucNDR_doubleNDR_2kb_100seq_flanksInt2_gclen100 \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nuc_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     --start_seqs /home/alex/SCerevisiae_chromatin_NN_prediction/generated/regnuc_2kb_100seq_randomflanks_gclen100/designed_seqs/mut_seqs_step499.npy \
#     --flanks /home/alex/shared_folder/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     --target_file /home/alex/SCerevisiae_chromatin_NN_prediction/generated/regnuc_2kb_100seq_randomflanks/target_with_NDR2.npz \
#     -gctol 0.01 -gclen 100 \
#     --steps 500 -t 0.0001 -s 16 --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/polsigmoid_right_2kb_100seq_flanksInt2 \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nucpol_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 -nt 2 -track 1 \
#     --flanks /home/alex/shared_folder/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     -gctol 0 -gclen 100 \
#     -n 100 -l 2000 --steps 500 -t 0.0001 -s 16 -ilen 500 -ishape sigmoid -bg low high --seed 0 -v

# python compute_shap_values.py \
#     -m Trainedmodels/model_myco_nuc_pt8/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw \
#     -o shap \
#     -arch BassenjiMultiNetwork2 \
#     -bc chrXVI -s for -mid -v

# arch=BassenjiMultiNetwork2
# model_dir=model_myco_nucpol_pt23
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o Trainedmodels/$model_dir -arch $arch \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 16384
# python predict_pytorch.py -m Trainedmodels/$model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o Trainedmodels/$model_dir -arch $arch -mid -v -nt 2 -w 16384 -b 512
# model_dir=model_myco_pol_pt23
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw \
#     -o Trainedmodels/$model_dir -arch $arch \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 16384
# python predict_pytorch.py -m Trainedmodels/$model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o Trainedmodels/$model_dir -arch $arch -mid -v -w 16384 -b 512
# model_dir=model_myco_nuc_pt23
# python train_pytorch.py -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -l /home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw \
#     -o Trainedmodels/$model_dir -arch $arch \
#     -ct I II III IV V VI VII VIII IX X XI XII XIII -cv XIV XV \
#     -bal -r0 -rN -v -loss mae_cor -w 16384
# python predict_pytorch.py -m Trainedmodels/$model_dir/model_state.pt \
#     -f /home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa \
#     -o Trainedmodels/$model_dir -arch $arch -mid -v -w 16384 -b 512

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/2kb_regnuc_lowpol_10seq_flanksInt2 \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nucpol_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -n 10 -l 2000 -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     --flanks /home/alex/shared_folder/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     --target_file /home/alex/SCerevisiae_chromatin_NN_prediction/generated/target4kb_lowpol_regnuc.npz \
#     -nt 2 -track 0 1 -twgt 1 1 \
#     -gctol 0.01 -gclen 100 \
#     --steps 500 -t 0.0001 -s 16 --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/2kb_regnuc_lowpol_10seq_flanksInt2_trackweight \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nucpol_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -n 10 -l 2000 -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     --flanks /home/alex/shared_folder/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     --target_file /home/alex/SCerevisiae_chromatin_NN_prediction/generated/target4kb_lowpol_regnuc.npz \
#     -nt 2 -track 0 1 -twgt 1 8 \
#     -gctol 0.01 -gclen 100 \
#     --steps 500 -t 0.0001 -s 16 --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/4kb_regnuc_lowpol_10seq_flanksInt2 \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nucpol_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -n 10 -l 4000 -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     --flanks /home/alex/shared_folder/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     --target_file /home/alex/SCerevisiae_chromatin_NN_prediction/generated/target4kb_lowpol_regnuc.npz \
#     -nt 2 -track 0 1 -twgt 1 8 \
#     -gctol 0.01 -gclen 100 \
#     --steps 1000 -t 0.0001 -s 16 --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/4kb+_regnuc_lowpol_10seq_randomflanks \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nucpol_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -n 10 -l 4008 -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     --flanks random \
#     --target_file /home/alex/SCerevisiae_chromatin_NN_prediction/generated/target4kb+_lowpol_regnuc.npz \
#     -nt 2 -track 0 1 -twgt 1 8 \
#     -gctol 0.01 -gclen 100 \
#     --steps 1000 -t 0.0001 -s 16 --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/4kb+_nucNDR_lowpol_10seq_randomflanks \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nucpol_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -n 10 -l 4175 -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     --flanks random \
#     --target_file /home/alex/SCerevisiae_chromatin_NN_prediction/generated/target4kb+_lowpol_nucNDR.npz \
#     -nt 2 -track 0 1 -twgt 1 8 \
#     -gctol 0.01 -gclen 100 \
#     --steps 1000 -t 0.0001 -s 16 --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/4kb+_regnuc_lowpol_10seq_randomflanks_gc0.2 \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nucpol_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -n 10 -l 4008 \
#     --flanks random \
#     --target_file /home/alex/SCerevisiae_chromatin_NN_prediction/generated/target4kb+_lowpol_regnuc.npz \
#     -nt 2 -track 0 1 -twgt 1 8 \
#     -gctol 0.01 -gclen 100 -targ_gc 0.2 \
#     --steps 1000 -t 0.0001 -s 16 --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/4kb+_nucNDR_lowpol_10seq_randomflanks_gc0.2 \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nucpol_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -n 10 -l 4175 \
#     --flanks random \
#     --target_file /home/alex/SCerevisiae_chromatin_NN_prediction/generated/target4kb+_lowpol_nucNDR.npz \
#     -nt 2 -track 0 1 -twgt 1 8 \
#     -gctol 0.01 -gclen 100 -targ_gc 0.2 \
#     --steps 1000 -t 0.0001 -s 16 --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/4kb+_regnuc_lowpol_10seq_randomflanks_gc0.2_twgt1_1 \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nucpol_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -n 10 -l 4008 \
#     --flanks random \
#     --target_file /home/alex/SCerevisiae_chromatin_NN_prediction/generated/target4kb+_lowpol_regnuc.npz \
#     -nt 2 -track 0 1 -twgt 1 1 \
#     -gctol 0.01 -gclen 100 -targ_gc 0.2 \
#     --steps 1000 -t 0.0001 -s 16 --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/4kb+_nucNDR_lowpol_10seq_randomflanks_gc0.2_twgt1_1 \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nucpol_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -n 10 -l 4175 \
#     --flanks random \
#     --target_file /home/alex/SCerevisiae_chromatin_NN_prediction/generated/target4kb+_lowpol_nucNDR.npz \
#     -nt 2 -track 0 1 -twgt 1 1 \
#     -gctol 0.01 -gclen 100 -targ_gc 0.2 \
#     --steps 1000 -t 0.0001 -s 16 --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/4kb+_regnuc_lowpol_10seq_randomflanks_twgt1_1 \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nucpol_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -n 10 -l 4008 -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     --flanks random \
#     --target_file /home/alex/SCerevisiae_chromatin_NN_prediction/generated/target4kb+_lowpol_regnuc.npz \
#     -nt 2 -track 0 1 -twgt 1 1 \
#     -gctol 0.01 -gclen 100 \
#     --steps 1000 -t 0.0001 -s 16 --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/2kb_regnuc_10seq_flanksInt2_gc0.4_nrl167 \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nuc_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -n 10 -l 2000 \
#     --flanks /home/alex/shared_folder/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#     -ilen 0 -per 167 -plen 147 -pshape gaussian \
#     -gctol 0.01 -gclen 100 -targ_gc 0.4 \
#     --steps 500 -t 0.0001 -s 16 --seed 0 -v

# for gc in 0.2 0.3 0.4 0.5 0.6 0.7
# do
#     for nrl in 165 167 169 171 173 175 177
#     do
#         python kMC_sequence_design_pytorch.py \
#             -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/2kb_regnuc_10seq_flanksInt2_gc$gc'_nrl'$nrl \
#             -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nuc_pt8/model_state.pt \
#             -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#             -n 10 -l 2000 \
#             --flanks /home/alex/shared_folder/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#             -ilen 0 -per $nrl -plen 147 -pshape gaussian \
#             -gctol 0.01 -gclen 100 -targ_gc $gc \
#             --steps 500 -t 0.0001 -s 16 --seed 0 -v
#     done
# done

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/4kb+_regnuc_lowpol_10seq_randomflanks_twgt1_1_v2 \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nucpol_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -n 10 -l 4008 -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     --flanks random \
#     --target_file /home/alex/SCerevisiae_chromatin_NN_prediction/generated/target4kb+_lowpol_regnuc.npz \
#     -nt 2 -track 0 1 -twgt 1 1 \
#     -gctol 0.01 -gclen 100 \
#     --steps 1000 -t 0.0001 -s 16 --seed 0 -v


# for gc in 0.2 0.3 0.4 0.5 0.6 0.7
# do
#     python kMC_sequence_design_pytorch.py \
#         -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/2kb_regnuc_10seq_flanksInt2_globalgc$gc'_'nrl167 \
#         -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nuc_pt8/model_state.pt \
#         -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#         -n 10 -l 2000 \
#         --flanks /home/alex/shared_folder/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#         -ilen 0 -per 167 -plen 147 -pshape gaussian \
#         -gctol 0.01 -gclen 4000 -targ_gc $gc \
#         --steps 1000 -t 0.0001 -s 16 --seed 0 -v
    
#     python kMC_sequence_design_pytorch.py \
#         -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/2kb_regnuc_10seq_flanksInt2_globalgc$gc'_'nrl167_gcwgt5 \
#         -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nuc_pt8/model_state.pt \
#         -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#         -n 10 -l 2000 \
#         --flanks /home/alex/shared_folder/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz \
#         -ilen 0 -per 167 -plen 147 -pshape gaussian \
#         -gctol 0.01 -gclen 4000 -targ_gc $gc --weights 5 1 1 1 \
#         --steps 1000 -t 0.0001 -s 16 --seed 0 -v
# done

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/4kb_regnuc_lowpol_10seq_randomflanks \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nucpol_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -n 10 -l 4000 -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     --flanks random \
#     --target_file /home/alex/SCerevisiae_chromatin_NN_prediction/generated/target4kb_lowpol_regnuc.npz \
#     -nt 2 -track 0 1 -twgt 1 1 \
#     -gctol 0.01 -gclen 100 \
#     --steps 1000 -t 0.0001 -s 16 --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/4kb_regnuc_lowpol_10seq_randomflanks_nrl197 \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nucpol_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -n 10 -l 4000 -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     --flanks random \
#     --target_file /home/alex/SCerevisiae_chromatin_NN_prediction/generated/target4kb_lowpol_regnuc_nrl197.npz \
#     -nt 2 -track 0 1 -twgt 1 1 \
#     -gctol 0.01 -gclen 100 \
#     --steps 1000 -t 0.0001 -s 16 --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/4kb_nucNDR_lowpol_10seq_randomflanks_nrl167_NDR200 \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nucpol_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -n 10 -l 4000 -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     --flanks random \
#     --target_file /home/alex/SCerevisiae_chromatin_NN_prediction/generated/target4kb_lowpol_nucNDR200.npz \
#     -nt 2 -track 0 1 -twgt 1 1 \
#     -gctol 0.01 -gclen 100 \
#     --steps 1000 -t 0.0001 -s 16 --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/4kb_nucNDR_lowpol_10seq_randomflanks_nrl167_NDR300 \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nucpol_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -n 10 -l 4000 -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     --flanks random \
#     --target_file /home/alex/SCerevisiae_chromatin_NN_prediction/generated/target4kb_lowpol_nucNDR300.npz \
#     -nt 2 -track 0 1 -twgt 1 1 \
#     -gctol 0.01 -gclen 100 \
#     --steps 1000 -t 0.0001 -s 16 --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/4kb_regnuc_lowpol_10seq_randomflanks_nrl171 \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nucpol_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -n 10 -l 4000 -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     --flanks random \
#     --target_file /home/alex/SCerevisiae_chromatin_NN_prediction/generated/target4kb_lowpol_regnuc_nrl171.npz \
#     -nt 2 -track 0 1 -twgt 1 1 \
#     -gctol 0.01 -gclen 100 \
#     --steps 1000 -t 0.0001 -s 16 --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/4kb_nucNDR_lowpol_10seq_randomflanks_nrl171_NDR250 \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nucpol_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -n 10 -l 4000 -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     --flanks random \
#     --target_file /home/alex/SCerevisiae_chromatin_NN_prediction/generated/target4kb_lowpol_nucNDR250_nrl171.npz \
#     -nt 2 -track 0 1 -twgt 1 1 \
#     -gctol 0.01 -gclen 100 \
#     --steps 1000 -t 0.0001 -s 16 --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/4kb_nucNDR_lowpol_10seq_randomflanks_nrl167_NDR250 \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nucpol_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -n 10 -l 4000 -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     --flanks random \
#     --target_file /home/alex/SCerevisiae_chromatin_NN_prediction/generated/target4kb_lowpol_nucNDR250.npz \
#     -nt 2 -track 0 1 -twgt 1 1 \
#     -gctol 0.01 -gclen 100 \
#     --steps 1000 -t 0.0001 -s 16 --seed 0 -v

# python kMC_sequence_design_pytorch.py \
#     -o /home/alex/SCerevisiae_chromatin_NN_prediction/generated/4kb_nucNDR_lowpol_10seq_randomflanks_nrl167_NDR350 \
#     -m /home/alex/SCerevisiae_chromatin_NN_prediction/Trainedmodels/model_myco_nucpol_pt8/model_state.pt \
#     -w 2048 -h_int 16 -arch BassenjiMultiNetwork2 -mid -b 1024 \
#     -n 10 -l 4000 -kfile /home/alex/shared_folder/SCerevisiae/genome/W303/W303_3mer_freq.csv \
#     --flanks random \
#     --target_file /home/alex/SCerevisiae_chromatin_NN_prediction/generated/target4kb_lowpol_nucNDR350.npz \
#     -nt 2 -track 0 1 -twgt 1 1 \
#     -gctol 0.01 -gclen 100 \
#     --steps 1000 -t 0.0001 -s 16 --seed 0 -v