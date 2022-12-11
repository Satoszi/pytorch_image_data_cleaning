# Image data cleaning with pytorch (GPU)

I created this small tool to clean a dataset from redundant images. It was no trivial process as redundant images often are not similar. F.e. they can have just a little different bightness, they can be cropped, mirrored etc.

This tool processes all images from a given directory by any choosen pretrained model (I used resnet) with cut last layer, to get latent vector. Then the tool uses this latent vecotr to compare image similarity (exactly cosine_similarity between vectors).

The vector comparison process is unfortunatelly n^2, because, every image vector must be comparised with each other. (There are algorithms for multi dimensional sorting, but I didn't have as many data, to struggle with that)
I used it for 7k images and it took 2-3h. 
For 1k images it takes 1-2min


# Examples

![](examples/examples_1.PNG)

More examples:
![](examples/examples_2.PNG)