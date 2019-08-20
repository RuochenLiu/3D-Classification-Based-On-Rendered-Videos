# 3D Classification Based On Rendered Videos
## Data
ModelNet10 dataset downloaded from [Princeton ModelNet](http://modelnet.cs.princeton.edu/ "Princeton ModelNet").
## Proposed Method
### Feature
- Original OFF files converted into PLY format.
- Each polygon object is rendered and taken images of from three orthogonal axes.
- Twelve images represent for each axis and are frames of one rendered video.
<img src="https://github.com/RuochenLiu/3D-Classification-Based-On-Rendered-Videos/blob/master/fig/sample_video/sample.gif" width="400" height="250">

### Network
- Plain 3D-CNN architecture with 16-filter layer stack.
- Batch normalization applied. LeakyReLU as activation function.
- Features from three axes are fed to the same network.
- Prediction is made by voting from three results.
- [Model summary (link)](https://github.com/RuochenLiu/3D-Classification-Based-On-Rendered-Videos/blob/master/doc/model_summary.txt "Model summary")
## Experiment
- Trained for 20 epochs.
- Best test accuracy **91.6%**.

![](https://github.com/RuochenLiu/3D-Classification-Based-On-Rendered-Videos/blob/master/fig/acc.png)![](https://github.com/RuochenLiu/3D-Classification-Based-On-Rendered-Videos/blob/master/fig/loss.png)
