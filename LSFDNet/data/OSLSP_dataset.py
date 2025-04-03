import torchvision.transforms
import glob
from torch.utils.data.dataset import Dataset
from scripts.util import *
from torchvision.transforms import functional as F
from basicsr.utils.registry import DATASET_REGISTRY
import core.metrics as Metrics
import numpy
import cv2

def pad_or_truncate(some_list, target_len):
    return some_list[:target_len] + [0,0,0,0]*(target_len - len(some_list))

def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.txt"))))
    data.sort()
    filenames.sort()
    return data, filenames

# 坐标转换，原始存储的是YOLOv5格式
# Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    
    for i in range(y.shape[0]):
        y[i, 0] = w if y[i, 0] > w else y[i, 0]
        y[i, 2] = w if y[i, 2] > w else y[i, 2]
        
        y[i, 1] = h if y[i, 1] > h else y[i, 1]
        y[i, 3] = h if y[i, 3] > h else y[i, 3]
    return y

def flip_pos(x,w):
    x = w/2 - (x - w/2)
    return x

@DATASET_REGISTRY.register()
class OSLSP_FusionDataset(Dataset):
    def __init__( self, opt):
        super(OSLSP_FusionDataset, self).__init__()
        assert opt["name"] in ['train', 'val', 'test'], 'name must be "train"|"val"|"test"'
        self.is_crop = opt["is_crop"]
        crop_size = opt["crop_size"]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor=1)
        self.dethr_transform = train_dethr_transform(crop_size)
        self.centerhr_transform = train_centerhr_transform(crop_size)
        self.min_max = (-1,1)
        self.is_size = opt["is_size"]
        self.is_pad = opt["is_pad"]
        self.hflip = torchvision.transforms.RandomHorizontalFlip(p=1.1)
        self.vflip = torchvision.transforms.RandomVerticalFlip(p=1.1)

        self.new_size_test = (128, 96)

        if opt["name"] == 'train':
            data_dir_SW = opt["SW_path"]
            data_dir_LW = opt["LW_path"]
            data_dir_pa = opt["paired_path"]
            label_dir = opt["label_path"]
            self.filepath_SW, self.filenames_SW = prepare_data_path(data_dir_SW)
            self.filepath_LW, self.filenames_LW = prepare_data_path(data_dir_LW)
            self.filepath_pa, self.filenames_pa = prepare_data_path(data_dir_pa)
            self.filepath_label, self.filenames_label = prepare_data_path(label_dir)
            self.split = opt["name"]
            self.length = min(len(self.filenames_SW), len(self.filenames_LW))

        elif opt["name"] == 'val':
            data_dir_SW = opt["SW_path"]
            data_dir_LW = opt["LW_path"]
            data_dir_pa = opt["paired_path"]
            self.filepath_SW, self.filenames_SW = prepare_data_path(data_dir_SW)
            self.filepath_LW, self.filenames_LW = prepare_data_path(data_dir_LW)
            self.filepath_pa, self.filenames_pa = prepare_data_path(data_dir_pa)
            self.split = opt["name"]
            self.length = min(len(self.filenames_SW), len(self.filenames_LW))

    def __getitem__(self, index):
        if self.split == 'train':
            SW_image = Image.open(self.filepath_SW[index])
            LW_image = Image.open(self.filepath_LW[index])
            pa_image = Image.open(self.filepath_pa[index])
            with open(self.filepath_label[index],"r") as f:    #设置文件对象
                lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
            h, w = SW_image.height, SW_image.width
            lb[:, 1:] = xywhn2xyxy(lb[:, 1:], w, h, 0, 0)  # 反归一化
            la = np.asarray(lb, dtype=int)
                                        

            if self.is_crop:
                crop_size = self.dethr_transform({'SW':SW_image,'label':la,'path':self.filepath_SW[index]})
                SW_image, LW_image, pa_image = F.crop(SW_image, crop_size[0], crop_size[1], crop_size[2],crop_size[3]), \
                                            F.crop(LW_image, crop_size[0], crop_size[1], crop_size[2], crop_size[3]), \
                                            F.crop(pa_image, crop_size[0], crop_size[1], crop_size[2], crop_size[3])
                object_label = crop_size[4]

            if self.is_size:
                pre_size, resize_size = test_size(SW_image.size)
                SW_image = SW_image.resize((resize_size[0], resize_size[1]),Image.LANCZOS) #resize image with high-quality
                LW_image = LW_image.resize((resize_size[0], resize_size[1]),Image.LANCZOS) #resize image with high-quality
                pa_image = pa_image.resize((resize_size[0], resize_size[1]),Image.LANCZOS) #resize image with high-quality
            
            # if self.is_pad:
            #     target_size = max(SW_image.width, SW_image.height)    
            #     # 创建新的正方形图像（使用指定的填充颜色）  
            #     result = Image.new('RGB', (target_size, target_size), (0, 0, 0))  #黑色
            #     # 计算粘贴位置（居中）  
            #     paste_x = (target_size - SW_image.width) // 2  
            #     paste_y = (target_size - SW_image.height) // 2  
                
            #     # 将原图粘贴到正方形画布上  
            #     result.paste(SW_image, (paste_x, paste_y)) 
            #     result.paste(LW_image, (paste_x, paste_y)) 
            #     result.paste(pa_image, (paste_x, paste_y)) 

            # Random horizontal flipping
            if random.random() > 0.5:
                SW_image = self.hflip(SW_image)
                LW_image = self.hflip(LW_image)
                pa_image = self.hflip(pa_image)
                for label in object_label:
                    label[1] = flip_pos(label[1],crop_size[3])
                    label[3] = flip_pos(label[3],crop_size[3])
                    label[1], label[3] = label[3], label[1]

            # Random vertical flipping
            if random.random() > 0.5:
                SW_image = self.vflip(SW_image)
                LW_image = self.vflip(LW_image)
                pa_image = self.vflip(pa_image)
                for label in object_label:
                    label[2] = flip_pos(label[2],crop_size[2])
                    label[4] = flip_pos(label[4],crop_size[2])
                    label[2], label[4] = label[4], label[2]
            #img=cv2.imread(self.filepath_LW[index])
            # LW_th = cv2.cvtColor(numpy.array(LW_image), cv2.COLOR_RGB2BGR)
            # n,LW_th=cv2.threshold(LW_th, 180, 200, cv2.THRESH_TOZERO)
            # LW_th=(cv2.cvtColor(LW_th,cv2.COLOR_BGR2RGB))  
            # LW_th = Image.fromarray(LW_th)
            # LW_th = ToTensor()(LW_th)*(self.min_max[1] - self.min_max[0]) + self.min_max[0]
            SW_image = ToTensor()(SW_image)*(self.min_max[1] - self.min_max[0]) + self.min_max[0]
            LW_image = ToTensor()(LW_image)*(self.min_max[1] - self.min_max[0]) + self.min_max[0]
            pa_image = ToTensor()(pa_image)*(self.min_max[1] - self.min_max[0]) + self.min_max[0]

            if len(object_label) > 10 :
                object_label = object_label[0:10]
            else:
                for i in range(len(object_label),10):
                    object_label.append(np.zeros(5).astype(int) )
            object_label = torch.IntTensor(np.array(object_label).astype(int))
            zero_img = torch.zeros_like(SW_image)

            # cat_img = torch.cat([SW_image[0:1, :, :], LW_image[0:1, :, :], zero_img[0:1, :, :]], axis=0)
            cat_img = torch.cat([SW_image[0:1, :, :], LW_image[0:1, :, :]], axis=0)
            # sl=Metrics.tensor2img(LW_image)
            # ss=Metrics.tensor2img(SW_image)
            # sp=Metrics.tensor2img(pa_image)
            # sc=Metrics.tensor2img(cat_img)
            # Metrics.save_img(sl, '/home/gyy/0fusion/exp_vis/%d_L.png'%index)
            # Metrics.save_img(ss, '/home/gyy/0fusion/exp_vis/%d_S.png'%index)
            # Metrics.save_img(sp, '/home/gyy/0fusion/exp_vis/%d_p.png'%index)
            # Metrics.save_img(sc, '/home/gyy/0fusion/exp_vis/%d_c.png'%index)

            # sc=Metrics.tensor2img(LW_th)
            # Metrics.save_img(sc, './exp_vis/pic1/%d_L.png'%index)

            # grid_img = Metrics.tensor2img(cat_img)
            # ttt = '{}/{}'.format('./exp_vis/pic', self.filenames_SW[index])
            # Metrics.save_img(grid_img, ttt)
            # img1 = cv2.imread(ttt)
            # for x in crop_size[4]:
            #     img1 = cv2.rectangle(img1, (int(x[1]), int(x[2])), (int(x[3]), int(x[4])), (255, 0, 0), 2)
            # #img1 = img1[i:i+th,j:j+tw]
            # cv2.imwrite('./exp_vis/pic/us_'+ttt.split('/')[-1],img1)

            # grid_img = Metrics.tensor2img(SW_image[0:1, :, :])
            # ttt = '/home/gyy/IRFusion-main/LSFDNet/test/pic/SW——{}'.format(self.filenames_SW[index])
            # Metrics.save_img(grid_img, ttt)
            # grid_img = Metrics.tensor2img(LW_image[0:1, :, :])
            # ttt = '/home/gyy/IRFusion-main/LSFDNet/test/pic/LW——{}'.format(self.filenames_SW[index])
            # Metrics.save_img(grid_img, ttt)

            return {'img': cat_img, 'SW': SW_image[0:1, :, :], 'LW': LW_image[0:1, :, :], 'label':object_label, 'LW_th':pa_image[0:1, :, :]},  self.filenames_SW[index]

        elif self.split == 'val':
            SW_image = Image.open(self.filepath_SW[index])
            LW_image = Image.open(self.filepath_LW[index])
            pa_image = Image.open(self.filepath_pa[index])
            ori_size = []
            ori_size.append(SW_image.width)
            ori_size.append(SW_image.height)
            ori_size = torch.tensor(ori_size)
            pre_size=[160,160]

            if self.is_crop:
                crop_size = self.centerhr_transform({'SW':SW_image})
                SW_image, LW_image, pa_image = F.crop(SW_image, crop_size[0], crop_size[1], crop_size[2],crop_size[3]), \
                                            F.crop(LW_image, crop_size[0], crop_size[1], crop_size[2], crop_size[3]), \
                                            F.crop(pa_image, crop_size[0], crop_size[1], crop_size[2], crop_size[3])

            if self.is_size:
                pre_size, resize_size = test_size(SW_image.size)
                SW_image = SW_image.resize((resize_size[0], resize_size[1]),Image.LANCZOS) #resize image with high-quality
                LW_image = LW_image.resize((resize_size[0], resize_size[1]),Image.LANCZOS) #resize image with high-quality
                pa_image = pa_image.resize((resize_size[0], resize_size[1]),Image.LANCZOS) #resize image with high-quality

            if self.is_pad:
                target_size = max(SW_image.width, SW_image.height)    
                # 创建新的正方形图像（使用指定的填充颜色）  
                result_S = Image.new('RGB', (target_size, target_size), (0, 0, 0))  #黑色
                result_L = Image.new('RGB', (target_size, target_size), (0, 0, 0))  #黑色
                result_p = Image.new('RGB', (target_size, target_size), (0, 0, 0))  #黑色
                # 计算粘贴位置（居中）  
                paste_x = (target_size - SW_image.width) // 2  
                paste_y = (target_size - SW_image.height) // 2  
                
                # 将原图粘贴到正方形画布上  
                result_S.paste(SW_image, (paste_x, paste_y)) 
                SW_image = result_S
                result_L.paste(LW_image, (paste_x, paste_y)) 
                LW_image = result_L
                result_p.paste(pa_image, (paste_x, paste_y)) 
                pa_image = result_p

            #
            SW_image = ToTensor()(SW_image)
            SW_image = SW_image*(self.min_max[1] - self.min_max[0]) + self.min_max[0]

            LW_image = ToTensor()(LW_image)
            LW_image = LW_image * (self.min_max[1] - self.min_max[0]) + self.min_max[0]

            pa_image = ToTensor()(pa_image)
            pa_image = pa_image * (self.min_max[1] - self.min_max[0]) + self.min_max[0]

            zero_img = torch.zeros_like(SW_image)
            # cat_img = torch.cat([SW_image[0:1, :, :], LW_image[0:1, :, :], zero_img[0:1, :, :]], axis=0)
            cat_img = torch.cat([SW_image[0:1, :, :], LW_image[0:1, :, :]], axis=0)

            #cat_img = torch.cat([SW_image[0:1, :, :], LW_image[0:1, :, :], pa_image[0:1, :, :]], axis=0)

            return {'img': cat_img, 'SW': SW_image[0:1, :, :], 'LW': LW_image[0:1, :, :], 'ori_size': ori_size},  self.filenames_SW[index]

    def __len__(self):
        return self.length
