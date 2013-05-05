# d should be attributes - 1, since one of the attributes is class label.
#time ./map_est --trainfile="../dataset/human_class5_train.dat" \
#--testfile="../dataset/human_class5_test.dat" \
#--s0=0.01 --d=561 --Ntrain=5881 --Ntest=1471 \
#--labelC1=5 \
#--max_iter=5000 \
#--save_binary=false

#time ./map_est --trainfile="../dataset/p53_train.dat" \
#--testfile="../dataset/p53_test.dat" \
#--s0=0.05 --d=5408 --Ntrain=28278 --Ntest=3142 \
#--labelC1=1 \
#--save_binary=false

#time ./map_est --trainfile="../dataset/farm_ads_train.dat" \
#--testfile="../dataset/farm_ads_test.dat" \
#--s0=0.05 --d=54877 --Ntrain=3314 --Ntest=829 \
#--labelC1=0 \
#--save_binary=false

# small test set
time ./map_est --trainfile="../dataset/human_class5_train.csv" \
--testfile="../dataset/human_class5_test.csv" \
--labelC1=5 --s0=0.05

#time ./map_est --trainfile="../dataset/human_class5_train_small.dat" \
#--testfile="../dataset/human_class5_test_small.dat" \
#--labelC1=0 \
#--s0=0.05 --d=54877 --Ntrain=500 --Ntest=200
