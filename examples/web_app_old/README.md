# CycleGAN Gradio Web App deployed on Heroku

This folder contains code to deploy a simple Horse-to-Zebra model onto Heroku for free. 

An example web app is running at: 
http://upit-cyclegan.herokuapp.com/

Note: it takes a couple minutes for the web app to load and a few seconds to produce a result. Upgrading to paid web apps (dynos) may alleviate some of these issues.

## How to deploy:

1. Install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli#download-and-install)
2. Clone this repository:
```
git clone https://github.com/tmabraham/UPIT.git
```
3. Initialize a Heroku web app from the command line (in the UPIT repository folder):
```
heroku create
```
4. Deploy the `web_app` folder 
```
git subtree push --prefix examples/web_app heroku master
```
And that's it! Heroku will process the code, compress it and deploy it, providing you with a link you can visit at any time!

This deployment was based on [this guide](https://towardsdatascience.com/how-to-deploy-a-machine-learning-ui-on-heroku-in-5-steps-b8cd3c9208e6).

## File descriptions:

`cyclegan_inference.ipynb` - This is a notebook with all the code to run a Gradio web app demo. Running this notebook will yield in a temporary temporary web app that is not persistent (runs for 6 hours). The `gradio.app` link is provided after running the code. Also note that this code is set up to run on the GPU. This can be easily changed by changing the `map_location` in `torch.load`.

`cyclegan_inference.py` - This is a Python file version of `cyclegan_inference.ipynb` that runs on the CPU and takes in 256x256 images (due to low memory for free Heroku dyno). This is what is used for the Heroku deployment.

`generator.pth` - My model file saved with the UPIT `export_generator` function. You can replace with your own generator.

`Procfile`,`requirements.txt`,`runtime.txt`,`setup.sh` - These files are required for Heroku deployment.
