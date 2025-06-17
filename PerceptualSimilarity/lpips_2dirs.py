import argparse
import os
import lpips
import torch.nn.functional as F
from DISTS_pytorch import DISTS
D = DISTS()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='./imgs/ex_dir0')
parser.add_argument('-d1','--dir1', type=str, default='./imgs/ex_dir1')
parser.add_argument('-o','--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
loss_fn = lpips.LPIPS(net='alex',version=opt.version)
if(opt.use_gpu):
	loss_fn.cuda()
	D.cuda()

# crawl directories
f = open(opt.out,'w')
files = os.listdir(opt.dir0)

l_lpips = 0
l_dists = 0
tot = 0

for file in files:
	if(os.path.exists(os.path.join(opt.dir1,file.replace(".png", "x3.png")))):
		# Load images
		img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
		img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,file.replace(".", "x3."))))

		if(opt.use_gpu):
			img0 = img0.cuda()
			img1 = img1.cuda()

		img1 = F.pad(img1, (0, img0.shape[-1] - img1.shape[-1], img0.shape[-2] - img1.shape[-2], 0), 'constant', 0)
		# Compute distance
		dist01 = loss_fn.forward(img0,img1)
		dists_value = D(img1, img0)
		f.writelines('%s: %.6f %.6f\n'%(file,dist01, dists_value))
        
		l_lpips += dist01.item()
		l_dists += dists_value.item()
		tot += 1

f.close()

print (f"LPIPS : {l_lpips / tot}")
print (f"DISTS : {l_dists / tot}")
