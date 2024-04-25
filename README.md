# README

This is a repository for SuperPoint-SuperPoint onnx & om models inference.

The C++ tensorrt inference source code is https://github.com/yuefanhao/SuperPoint-SuperGlue-TensorRT.

We convert and inference the onnx & om models according to the code in the above repository.

For the om model conversion method, see the following code

```bash
# superpoint
atc --model=superpoint_v1.onnx --framework=5 --output=superpoint --input_shape="input:1,1,480,320"  --soc_version=Ascend310
# superglue
atc --model=superglue_outdoor.onnx --framework=5 --output=superglue --input_shape="keypoints_0:1,512,2;scores_0:1,512;descriptors_0:1,256,512;keypoints_1:1,512,2;scores_1:1,512;descriptors_1:1,256,512"  --soc_version=Ascend310
```


