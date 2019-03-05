import os
import sys
os.system('python vsqx2npy.py %s out.npy'%sys.argv[1])
os.system('python synthesis.py checkpoint_step000800000_ema.pth .\out --conditional="out.npy"')
os.system('python waveglow_generate.py --checkpoint="model.ckpt-660000.pt" --local_condition_file=".\out\checkpoint_step000800000_ema.npy"')