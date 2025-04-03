import sys  
import os  
script_directory =os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 脚本目录的绝对路径  
sys.path.append(script_directory)  
import torch  
import torch.nn as nn  
from thop import profile
from archs.MMFusion_arch import MMFusion
import math
# 创建组合模型实例  
big_model = MMFusion()

# 要生成适当的输入数据  
input_tensor1 = torch.randn(1, 1, 320, 320)
input_tensor2 = torch.randn(1, 1, 320, 320)
input_tensor = {'SW':input_tensor1, 'LW':input_tensor2}

# 使用 thop 计算整体 FLOPs 和参数量  
flops, params = profile(big_model, inputs=(input_tensor,))  

print(f"组合模型的浮点运算量为: {flops / 1e9:.2f} GFLOPs")  # 转换为 GFLOPs  
print(f"组合模型的参数总量为: {params} 个参数")  