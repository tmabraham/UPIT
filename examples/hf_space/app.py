from fastai.vision.all import *
from fastai.basics import *
from upit.models.cyclegan import *
from upit.train.cyclegan import *
from upit.data.unpaired import *
import torchvision
import gradio as gr

dls = get_dls_from_hf("huggan/horse2zebra", load_size=286)
cycle_gan = CycleGAN.from_pretrained('tmabraham/horse2zebra_cyclegan')

totensor = torchvision.transforms.ToTensor()
normalize_fn = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
topilimage = torchvision.transforms.ToPILImage()

model = cycle_gan.G_B.cpu().eval()
def predict(input):
    im = normalize_fn(totensor(input))
    print(im.shape)
    preds = model(im.unsqueeze(0))/2 + 0.5
    print(preds.shape)
    return topilimage(preds.squeeze(0).detach())

gr_interface = gr.Interface(fn=predict, inputs=gr.inputs.Image(shape=(256, 256)), outputs="image", title='Horse-to-Zebra CycleGAN with UPIT', description='[This](https://huggingface.co/tmabraham/horse2zebra_cyclegan) CycleGAN model trained on [this dataset](https://huggingface.co/datasets/huggan/horse2zebra), using the [UPIT package](https://github.com/tmabraham/UPIT)', examples=['horse.jpg'])
gr_interface.launch(inline=False,share=False)
