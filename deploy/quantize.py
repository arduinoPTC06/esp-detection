import os
from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from onnxsim import simplify
import onnx


class CaliDataset(Dataset):
    def __init__(self, path, img_shape=640):
        super().__init__()
        height, width = img_shape if isinstance(img_shape, (list, tuple)) else (img_shape, img_shape)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((height, width)),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ]
        )

        self.imgs_path = []
        self.path = path
        for img_name in os.listdir(self.path):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            img_path = os.path.join(self.path, img_name)
            self.imgs_path.append(img_path)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img = Image.open(self.imgs_path[idx])
        img = self.transform(img)
        return img


def report_hook(blocknum, blocksize, total):
    downloaded = blocknum * blocksize
    percent = downloaded / total * 100
    print(f"\rDownloading calibration dataset: {percent:.2f}%", end="")


def quant_espdet(onnx_path, target, num_of_bits, device, batchsz, imgsz, calib_dir, espdl_model_path):
    INPUT_SHAPE = [3, *imgsz] if isinstance(imgsz, (list, tuple)) else [3, imgsz, imgsz]
    model = onnx.load(onnx_path)
    sim = True
    if sim:
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx.shape_inference.infer_shapes(model), onnx_path)

    calibration_dataset = CaliDataset(calib_dir, img_shape=imgsz)
    dataloader = DataLoader(
        dataset=calibration_dataset, batch_size=batchsz, shuffle=False
    )

    def collate_fn(batch: torch.Tensor) -> torch.Tensor:
        return batch.to(device)

    # default setting
    quant_setting = QuantizationSettingFactory.espdl_setting()

    # Equalization
    quant_setting.equalization = True
    quant_setting.equalization_setting.iterations = 4
    quant_setting.equalization_setting.value_threshold = .4
    quant_setting.equalization_setting.opt_level = 2
    quant_setting.equalization_setting.interested_layers = None


    quant_ppq_graph = espdl_quantize_onnx(
        onnx_import_file=onnx_path,
        espdl_export_file=espdl_model_path,
        calib_dataloader=dataloader,
        calib_steps=32,
        input_shape=[1] + INPUT_SHAPE,
        target=target,
        num_of_bits=num_of_bits,
        collate_fn=collate_fn,
        setting=quant_setting,
        device=device,
        error_report=True,
        skip_export=False,
        export_test_values=False,
        verbose=0,
        inputs=None,
    )
    return quant_ppq_graph  # , selected


if __name__ == "__main__":
    quant_espdet(
        onnx_path="espdet_pico_224_224_cat.onnx",
        target="esp32p4",
        num_of_bits=8,
        device='cpu',
        batchsz=32,
        imgsz=224,
        calib_dir="deploy/cat_calib",
        espdl_model_path="espdet_pico_224_224_cat.espdl",
    )
