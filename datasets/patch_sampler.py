# datasets/patch_sampler.py
import random
from PIL import ImageStat

class VarianceBiasedCrop:
    """
    基于信息量方差的图块选择模块 (Informative Patch Selection)
    用于在 1K 高清大图上提取包含丰富纹理和失真细节的高频图块。
    """
    def __init__(self, size=224, num_patches=1, candidate_pool_size=10, mode='train'):
        self.size = size
        self.num_patches = num_patches
        # 候选池的大小必须大于等于需要提取的图块数
        self.candidate_pool_size = max(candidate_pool_size, num_patches)
        self.mode = mode

    def __call__(self, img):
        w, h = img.size
        candidates = []
        
        # 1. 随机生成大量候选图块 (Candidate Pool)
        for _ in range(self.candidate_pool_size):
            i = random.randint(0, max(0, h - self.size))
            j = random.randint(0, max(0, w - self.size))
            patch = img.crop((j, i, j + self.size, i + self.size))
            
            # 2. 计算强度方差 (Patch Intensity Variance, PIV)
            # 转为灰度图计算方差，方差越低代表区域越平滑（如纯色天空）
            stat = ImageStat.Stat(patch.convert('L'))
            variance = stat.var[0]
            candidates.append((variance, patch))
        
        # 3. 按方差从大到小排序，剔除低方差平滑区域
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        if self.mode == 'train':
            # 训练阶段：为了防止模型过拟合最复杂的区域，我们在前 50% 的高方差图块中随机挑选
            top_k = max(self.num_patches, self.candidate_pool_size // 2)
            selected = random.sample(candidates[:top_k], self.num_patches)
        else:
            # 验证/测试阶段：直接选取方差最高的固定数量图块，确保评估结果稳定且针对失真区域
            selected = candidates[:self.num_patches]
        
        # 返回挑选好的 PIL Image 列表
        return [patch for var, patch in selected]
