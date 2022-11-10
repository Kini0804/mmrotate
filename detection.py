from mmrotate.apis import init_detector, inference_detector_by_patches
import mmcv
import json
from PIL import Image


class Detection():

    def __init__(self, config_file):
        configs = json.load(open(config_file, "r"))
        config_file = configs['model_structure']
        checkpoint_file = configs['model_weights']
        configs['use_gpu'] = True if configs['use_gpu']=="true" else False
        self.model = init_detector(config_file, checkpoint_file, device='cpu' if not configs['use_gpu'] else 'cuda:0')

    def detect(self, img):
        # 测试单张图片并保存结果
        result = inference_detector_by_patches(self.model, img) # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
        # 在一个新的窗口中将结果可视化
        # model.show_result(img, result)
        # 或者将可视化结果保存为图片
        result_image_arr = self.model.show_result(img, result)
        im = Image.fromarray(result_image_arr)
        im.save("tmp/result.jpg")
        # return result_image_arr