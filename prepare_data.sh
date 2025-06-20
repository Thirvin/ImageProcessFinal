echo "preparing dataset..."
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X3.zip -P datat
unzip datat/DIV2K_train_LR_bicubic_X3.zip  -t datat

wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip -P datat
unzip datat/DIV2K_train_HR.zip -t datat

python scripts/png2npy.py --pathFrom datat --pathTo datatt


wget https://huggingface.co/datasets/eugenesiow/BSD100/resolve/main/data/BSD100_HR.tar.gz -P eval_data/
wget https://huggingface.co/datasets/eugenesiow/BSD100/resolve/main/data/BSD100_LR_x3.tar.gz -P eval_data/

unzip eval_data/BSD100_HR.tar.gz -t eval_data/BSD100
unzip eval_data/BSD100_LR_x3.tar.gz -t eval_data/BSD100


wget https://huggingface.co/datasets/eugenesiow/Set14/resolve/main/data/Set14_HR.tar.gz -P eval_data/
wget https://huggingface.co/datasets/eugenesiow/Set14/resolve/main/data/Set14_LR_x3.tar.gz -P eval_data/

unzip eval_data/Set14_HR.tar.gz -t eval_data/Set14
unzip eval_data/Set14_LR_x3.tar.gz -t eval_data/Set14

echo "done"
