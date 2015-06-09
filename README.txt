MorphLib
==============

This C++ package is a simplified Version (pure c++ Library) for
"Automating Image Morphing using Structural Similarity on a Halfway Domain （Siggraph 2014)"

It has been successfully compiled in the x64 Windows with the compiler Visual Studio 2013. The CPUMorph Library has no other dependency. While the example project which show just how to use the CPUMorph Library requires the OpenCV Library.

Those Simplifications include:
(1)	Remove GPU acceleration
(2)	Remove quadratic path
(3)	Remove Poisson boundary extension
(4)	Use 3x3 ssim neighbors to replace 5x5 neighbors

The full version can be found here:
https://github.com/liaojing/Image-Morphing/

and the executable program is:
https://drive.google.com/folderview?id=0BwMKxLMS8dFBSTBPa2lRUWxGbFk&usp=sharing

Jing Liao

2015/06/01
