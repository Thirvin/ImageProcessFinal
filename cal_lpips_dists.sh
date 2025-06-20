cd PerceptualSimilarity 
python lpips_2dirs.py -d0 ../val/Set14_x2/Set14_HR -d1 ../results/aesop_Set14x3/ -o aesop_set14.txt --use_gpu
python lpips_2dirs.py -d0 ../val/bsd100/BSD100_HR -d1 ../results/aesop_BSD100x3/ -o aesop_BSD100.txt --use_gpu
python lpips_2dirs.py -d0 ../val/Set14_x2/Set14_HR -d1 ../results/Set14x3 -o baseline_set14.txt --use_gpu
python lpips_2dirs.py -d0 ../val/bsd100/BSD100_HR -d1 ../results/BSD100x3 -o baseline_BSD100.txt --use_gpu
