from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision  import transforms

class ImageFolderDataset(Dataset):
    def __init__(self, root, transform):
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform

        self.files=list(os.listdir(root))
        self.files = [p for p in self.files if p.endswith(('.jpg', '.png', 'jpeg'))]


    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.files[idx])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


def get_transform(size, crop, final_size):

    transform_list = []



    if crop:
        transform_list.append(transforms.Resize(size))
        transform_list.append(transforms.RandomCrop(final_size))

    else:
        transform_list.append(transforms.Resize((final_size, final_size)))

    transform_list.append(transforms.ToTensor()) # 
    return transforms.Compose(transform_list)   # Pipeline

def adaptive_instance_normalization(c_feats, s_feats):
    #  intput [batchsize, channel, h, w]
    size = c_feats.size()

    style_mean, style_std = cal_mean_std(s_feats)
    content_mean, content_std = cal_mean_std(c_feats)
    normalized_content_feat = (c_feats - content_mean.expand(size))/ content_std.expand(size)
    return normalized_content_feat * style_std.expand(size) + style_mean.expand(size)

def cal_mean_std(feats, eps=1e-5):
    #  intput [batchsize, channel, h, w]
    size = feats.size()

    assert (len(size) == 4)

    batch_size, channels = size[:2]
    feat_mean = feats.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1,1)
    feat_var = feats.view(batch_size, channels, -1).var(dim=2,unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1,1)

    return feat_mean, feat_std

    
