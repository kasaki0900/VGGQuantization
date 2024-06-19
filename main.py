from trans.de_quantization import *
from trans.quantization import *
from trans import pre_clip

if __name__ == '__main__':
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)

    qt_path = "weights/uint16_VGG16_cifar10_8904.pth"
    pth_path = "weights/VGG16_cifar10_8931.pth"
    model = VGG16.form_model()
    _, test_dataset = data.form_datasets()
    test_loader = data.form_dataloader(
        test_dataset,
        batch_size=64,
        test=True
    )

    # 解压缩并测试
    de_qt = LayerDeQuantizationTool(qt_path)

    model.load_state_dict(de_qt.de_quantize())
    acc = evaluate(model, test_loader, 'cuda')
    print(f'accuracy: {acc}%')
    # print(de_qt.de_quantize())


