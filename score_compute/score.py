from cleanfid import fid
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


fdir1 = "/home/pengjie/SiT_0/sample_results/test_dresscode/"
fdir2 = "/home/pengjie/SiT_0/sample_results/dc3_20000d/" 
score = fid.compute_fid(fdir1, fdir2, mode="clean")
print("fid:",score)

score = fid.compute_kid(fdir1, fdir2, mode="clean")
print("kid:",score)