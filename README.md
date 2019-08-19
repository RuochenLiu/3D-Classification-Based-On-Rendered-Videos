# 3D Classification Based On Rendered Videos
## Data
ModelNet10 dataset downloaded from [Princeton ModelNet](http://modelnet.cs.princeton.edu/ "Princeton ModelNet").
## Purposed Method
### Feature
Original OFF files are converted to PLY format. Each polygon object is rendered and taken images of by camera from three orthogonal axes as x-y-z. Twelve images represent for each axis and are frames of one rendered video.
![sample](https://github.com/RuochenLiu/3D-Classification-Based-On-Rendered-Videos/blob/master/fig/sample_video/sample.gif "sample")
### Network
