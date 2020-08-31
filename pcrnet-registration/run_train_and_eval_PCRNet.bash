#!/bin/bash

nvidia-smi

cd $PWD/utils/pc_distance/
# Change name to makefile_8.0 if you have CUDA-8.0 and Ubuntu-14.04
# If it down't work, get into `makefile_10.0_3.7` file and update tf path.
make -f makefile_10.0_3.7
cd $PWD/../..

PY="python3"
# Prefer python3
# For python2.7, follow steps mentioned below:
# Open train_iPCRNet.py, test_iPCRNet.py, statistical_analysis.py
# 	a. Remove "from numpy import matlib as npm"
# 	b. Replace "npm" with "np.matlib"
SPARSE=1
NUM_P=64
CUDA=0
EMD=1
CHAMFER=1
OURS=1
lim_rot=45.0
lim=45
MAX_EROR=45
PN=1
pn_pool="max"
MODE="train"
RESULTS="best_model"
NOISE=0
centroid_sub=0 #default=1
train_single=0 #default=0
iterations=8 #default=8
out_features=1024 #default=1024
template_random_pose=0
#    SPARSE=0 #SPARSE_SAMPLING
OCCLUSIONS=0.0
OCC=''


MODEL_WITH_NOISE=1 #dpdist model type (doesn't add noise, just has the option to add)
CAT='chair'

DATA=$CAT'_data'
EX2="new_ex41_sparse"$SPARSE""$OCC"_samples"$template_random_pose"_sin"$iterations"_"$MAX_EROR"1_"$CAT"ep_rot_lim"$lim"pn"$PN"pool"$pn_pool""$out_features
TEST_DATA=$CAT'_data_test'

# Choose poses file according to the wanted max transformation (rotation)
TRAIN_P='itr_net_train_data'$MAX_EROR'_1.csv'
EVAL_P='itr_net_eval_data'$MAX_EROR'_1.csv'
TEST='itr_net_test_data'$MAX_EROR'_1.csv'

# Choose dpdist trained model folder
dpdist_model="../log/test2__imp_type_1enc_3dmfvEmbS512BN0LR0001wd0sigma125K5AUG1lsl1_distnoise0chairov1np"$NUM_P"/"


EX_NAME=$EX2
mkdir $EX_NAME
mkdir "$EX_NAME/log_itrPCRNet_ours_$NUM_P"
cp $dpdist_model"/model.ckpt"* "$EX_NAME/log_itrPCRNet_ours_$NUM_P/"

SAMPLES=1.0 # A and B are **different** samples of the same surface
#
## Train PCRNet using (1) EMD, (2) Chamfer, and (3) DPDist (remove the & if there is not enough space on your GPU to train all three together)
#LOG_DIR="$EX_NAME/log_itrPCRNet_emd_$NUM_P"		# Folder name to store log.4
#CUDA_VISIBLE_DEVICES=0 $PY iterative_PCRNet.py -log $LOG_DIR -mode $MODE -results $RESULTS -noise $NOISE  --loss_type emd --num_point $NUM_P --s_random_points $SAMPLES --data_dict $DATA --train_single=$train_single --iterations $iterations --centroid_sub $centroid_sub --train_poses $TRAIN_P --eval_poses $EVAL_P --pointnet=$PN --lim_rot=$lim_rot --pn_pool=$pn_pool --out_features=$out_features --template_random_pose $template_random_pose --SPARSE_SAMPLING $SPARSE --add_occlusions $OCCLUSIONS &
#
#LOG_DIR="$EX_NAME/log_itrPCRNet_chamf_$NUM_P"		# Folder name to store log.4
#CUDA_VISIBLE_DEVICES=0 $PY iterative_PCRNet.py -log $LOG_DIR -mode $MODE -results $RESULTS -noise $NOISE  --loss_type chamf --num_point $NUM_P --s_random_points $SAMPLES --data_dict $DATA --train_single=$train_single --iterations $iterations --centroid_sub=$centroid_sub --train_poses $TRAIN_P --eval_poses $EVAL_P --pointnet=$PN --lim_rot=$lim_rot --pn_pool=$pn_pool --out_features=$out_features --template_random_pose $template_random_pose --SPARSE_SAMPLING $SPARSE --add_occlusions $OCCLUSIONS &
#
#LOG_DIR="$EX_NAME/log_itrPCRNet_ours_$NUM_P"		# Folder name to store log.4
#CUDA_VISIBLE_DEVICES=1 $PY iterative_PCRNet_ours.py -log $LOG_DIR -mode $MODE -results $RESULTS -noise $NOISE  --num_point $NUM_P --s_random_points $SAMPLES --data_dict $DATA --train_single=$train_single --iterations $iterations --centroid_sub $centroid_sub --train_poses $TRAIN_P --eval_poses $EVAL_P --model_with_noise $MODEL_WITH_NOISE --pointnet=$PN --lim_rot=$lim_rot --pn_pool=$pn_pool --out_features=$out_features --template_random_pose $template_random_pose --SPARSE_SAMPLING $SPARSE --add_occlusions $OCCLUSIONS

wait

for TEST_ITR in 8
do
    # Extract test results, open  `text.txt` in each folder to view the results.
    EX_NAME=$EX2
    SAMPLES=1.0
    #results:
    LOG_DIR="$EX_NAME/log_itrPCRNet_emd_$NUM_P"		# Folder name to store log.4
    WEIGHTS=$PWD/$LOG_DIR/$RESULTS".ckpt"
    LOG_RESULTS="$EX_NAME/results_"$TEST_ITR"itrPCRNet_emd_$NUM_P"
    # Test iterative PCRNet
    CUDA_VISIBLE_DEVICES=$CUDA $PY results_itrPCRNet_no_stop.py -weights $WEIGHTS -log $LOG_RESULTS -noise $NOISE  --num_point $NUM_P --s_random_points $SAMPLES --data_dict $TEST_DATA --centroid_sub $centroid_sub  --eval_poses $TEST --iterations $TEST_ITR --pointnet=$PN --lim_rot=$lim_rot --pn_pool=$pn_pool --out_features=$out_features  --template_random_pose $template_random_pose --SPARSE_SAMPLING $SPARSE --add_occlusions $OCCLUSIONS

    LOG_DIR="$EX_NAME/log_itrPCRNet_chamf_$NUM_P"
    WEIGHTS=$PWD/$LOG_DIR/$RESULTS".ckpt"
    LOG_RESULTS="$EX_NAME/results_"$TEST_ITR"itrPCRNet_chamf_$NUM_P"
    # Test iterative PCRNet
    CUDA_VISIBLE_DEVICES=$CUDA $PY results_itrPCRNet_no_stop.py -weights $WEIGHTS -log $LOG_RESULTS -noise $NOISE  --num_point $NUM_P --s_random_points $SAMPLES --data_dict $TEST_DATA --centroid_sub $centroid_sub  --eval_poses $TEST --iterations $TEST_ITR --pointnet=$PN --lim_rot=$lim_rot --pn_pool=$pn_pool --out_features=$out_features --template_random_pose $template_random_pose --SPARSE_SAMPLING $SPARSE --add_occlusions $OCCLUSIONS

    LOG_DIR="$EX_NAME/log_itrPCRNet_ours_$NUM_P"
    WEIGHTS=$PWD/$LOG_DIR/$RESULTS".ckpt"
    LOG_RESULTS="$EX_NAME/results_"$TEST_ITR"itrPCRNet_our_$NUM_P"
    # Test iterative PCRNet
    CUDA_VISIBLE_DEVICES=$CUDA $PY results_itrPCRNet_no_stop.py -weights $WEIGHTS -log $LOG_RESULTS -noise $NOISE  --num_point $NUM_P --s_random_points $SAMPLES --data_dict $TEST_DATA --centroid_sub $centroid_sub  --eval_poses $TEST --iterations $TEST_ITR --pointnet=$PN --lim_rot=$lim_rot --pn_pool=$pn_pool --out_features=$out_features --template_random_pose $template_random_pose --SPARSE_SAMPLING $SPARSE --add_occlusions $OCCLUSIONS
done

