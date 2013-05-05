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
#time ./main_binary --trainfile="../dataset/human_class5_train.csv" \
#--testfile="../dataset/human_class5_test.csv" \
#--labelC1=5 --s0=0.05 --algo="delta"

# yeast
time ./main_yeast --trainfile="../dataset/yeast-train.csv" \
--testfile="../dataset/yeast-test.csv" \
--labelC1=5 --s0=0.05 --max_iter=500 --algo="delta"
