# SegmentTrack
Algorithm to segment and track selected areas on a video. Using OpenCV

## Presentation
This program allows one to choose on-the-fly, zones in a video to segment and track and also to switch the algorithm segmentation mode, tracking mode, pause the program and more. It works using descriptors/features of extracted edges of the image and matching those features inter-frame. This program could have applications in robot vision, camera stabilization, scene-awareness for self-guidance, surveillance systems and so on. 

## Walkthrough
The input can be a recorded video or the webcam. In the following examples, the inputs are pre-recorded mock surgery videos, as if there is a medical robot watching the scene.

It starts by pre-processing the frame. Enhanced Local Contrast (CLAHE) and median filter is used. 

![Picture1](https://user-images.githubusercontent.com/44913276/75934638-d104a980-5e5b-11ea-8561-b147d73b655e.png)



Next step is segmentation. Canny or Adaptive Threshold can be choosed. 
Here is the result after using Adaptive Threshold and removing smaller blobs.

![Picture2](https://user-images.githubusercontent.com/44913276/75935178-4de45300-5e5d-11ea-8ce0-8458b574e101.png)



Then, the contour of each blob is extracted. Each contour will have a centroid (in red) and 3 descriptors. 
By comparing the descriptors, the algorithm will look for the same contour of the previous frame in the next frame and match them by similary.
The first contour on the list that passes the 3 similarity criteria will be chosen as the new tracked contour. 

The descriptors are:

 1. Distance from the candidate centroid to the previous tracked contour's centroid.
 2. Mean distance of 10 equally spaced points in the contour to its centroid location.
 3. Number of pixels in the contour.

Here is an image with the extracted contours, along with their indexes and 3 descriptors.

![Picture2 5](https://user-images.githubusercontent.com/44913276/75935729-e0d1bd00-5e5e-11ea-8767-a023deabf831.png)



Once the user selects which contour to track by placing the purple circle next to a centroid, the selected contour will be tracked and laid over the original frame, segmenting an area with high contrast.

![Picture3](https://user-images.githubusercontent.com/44913276/75935765-f2b36000-5e5e-11ea-9fd2-18b2a9019997.png)

![Picture4](https://user-images.githubusercontent.com/44913276/75936749-d238d500-5e61-11ea-925b-ac347e6ac32b.png)



In case the edges in the video are harder to track, there is a functionality that might help. It stills choses a centroid to track and to feature-match, but it lays all contours that have centroids within a radius on the final frame. 
To perform well, the chosen centroid must be high contrast and stable edge (a landmark). 
This is an example using a fake kidney image. 

![Picture5 5](https://user-images.githubusercontent.com/44913276/75937592-204ed800-5e64-11ea-9dbe-9e9cbdf4d58c.png)

![Picture6 5](https://user-images.githubusercontent.com/44913276/75938667-1a0e2b00-5e67-11ea-93b0-6c64247bc470.png)



As mentioned before, the segmentation step can be toggled between Canny Algorithm and Adaptive Threshold. 
Using Canny, the low and high threshold of the canny algorithm can be changed in real time through the slide bars.
By so, the segmentation result over the frame also change.

![Picture8](https://user-images.githubusercontent.com/44913276/75938220-f8f90a80-5e65-11ea-9fca-25999b9fac3d.png)



Once the algorithm is set to run, the user can interact with program, toggle the segmentation mode, radius mode, pause the program and so on. A list of the hotkeys to press is in the Hotkeys page.

