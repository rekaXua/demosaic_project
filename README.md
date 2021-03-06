# demosaic_project
Removing Pixelated Mosaic Censorship using [ESRGAN (Enhanced SRGAN)](https://github.com/xinntao/ESRGAN) and [green_mask_project](https://github.com/rekaXua/green_mask_project)

**ESRGAN model should be downloaded here: Twittman's [4x_FatalPixels_340000_G.pth](https://de-next.owncube.com/index.php/s/mDGmi7NgdyyQRXL/download?path=%2F&files=4x_FatalPixels_340000_G.pth), and placed into models folder!**

[![GitHub issues](https://img.shields.io/github/issues/rekaXua/demosaic_project.svg?label=Issues)](https://github.com/rekaXua/demosaic_project/issues)
[![GitHub downloads](https://img.shields.io/github/downloads/rekaXua/demosaic_project/total.svg?label=Downloads)](https://github.com/rekaXua/demosaic_project/releases)
[![GitHub release](https://img.shields.io/github/release/rekaXua/demosaic_project.svg?label=Version)](https://github.com/rekaXua/demosaic_project/releases/latest)
[![Twitter Follow](https://img.shields.io/twitter/follow/Alexander_rekaX.svg?label=Alexander_rekaX&style=flat&logo=twitter)](https://twitter.com/Alexander_rekaX/)
[![Donate with PayPal](https://img.shields.io/badge/PayPal-Donate-gray.svg?logo=paypal&label=)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=PDS9QQPVNUERE)

To have a decent performance you need a **[CUDA](https://developer.nvidia.com/cuda-toolkit-archive)+[cudnn](https://developer.nvidia.com/rdp/cudnn-download)(NVIDIA) or [ROCm](https://rocm.github.io/languages.html)(AMD) compatible GPU** with installed drivers! Otherweis  use "--cpu" key, but beware it can be VERY slow.

**HOW TO USE:**
1) Place your images inside decensor_input
2) Run "python demosaic_project_ESRGAN.py" or ("python demosaic_project_ESRGAN.py --cpu" for cpu version)
3) Take your decensored images from decensor_output
4) ???
5) Profit

**Don't forget to install python3 and all the requirements with command "pip3 install -r requirements.txt" in cmd or bash**
<p align="center">
  <img src="https://github.com/rekaxua/demosaic_project/blob/master/decensor_input/asuka.jpg" width="400">
  <img src="https://github.com/rekaxua/demosaic_project/blob/master/decensor_output/asuka.jpg" width="400">
</p>

**TODO:**
- GUI
- Config file
- Better detection algorithm  
  
Inspiration from [DeepCreamPy](https://github.com/deeppomf/DeepCreamPy) and [hent-AI](https://github.com/natethegreate/hent-AI)  
Credits to [Twittman](https://github.com/alsa64/AI-wiki/wiki/Model-Database) for making trained model [Fatal Pixels](https://de-next.owncube.com/index.php/s/mDGmi7NgdyyQRXL).  
  
*Sample image by hks(@timbougami)*
