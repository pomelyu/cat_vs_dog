import numpy as np
import torch
from tqdm import tqdm

from models import ResNet34
from dataset import dataloader
from config import TestOptions

def test(model, test_datatloader, opt):
    model.eval()

    res = []
    for _, (data, img_id) in tqdm(enumerate(test_datatloader), total=len(test_datatloader), ascii=True):
        if opt.use_gpu:
            data = data.cuda()

        score = model(data)
        prediction = (score >= 0.5).float()
        if opt.use_gpu:
            prediction = prediction.cpu()
        
        id_pred = np.stack([img_id.numpy(), prediction.detach().numpy()], axis=1)
        res.append(id_pred)

    res = np.concatenate(res, axis=0)
    return res


if __name__ == "__main__":
    opt = TestOptions().parse()

    test_datatloader = dataloader.create_test_dataloader(opt.test_data_path, opt.batch_size)

    model = ResNet34()
    model.load(opt.load_model_path)

    if opt.use_gpu:
        model = model.cuda()

    res = test(model, test_datatloader, opt)

    with open(opt.output_path, "w+") as fout:
        fout.write("id,label\n")
        for image_id, label in res:
            fout.write("{:.0f},{:.0f}\n".format(image_id, label))
