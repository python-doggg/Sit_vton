import torch_fidelity
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

metrics_dict = torch_fidelity.calculate_metrics(
    input1="/home/pengjie/SiT_0/sample_results/test_dresscode_256_192/", 
    input2="/home/pengjie/SiT_0/sample_results/dc3_20000d_256_192/", 
    cuda=True,  
    fid=True, 
    kid=True, 
    verbose=False,
)

print(metrics_dict)