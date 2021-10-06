# Interactive Billard (Also know as Pool Table)
 
The Interative Billard will project useful informations (such as speed, acceleration...) directly on the board.

https://user-images.githubusercontent.com/82355033/136162758-bdefb44f-0264-434b-a80b-5c885dc3d04b.mp4

## Requirement

You need to have at least a **Billard**, a **Webcam** and a **Video Projector**

Your webcam must be connect to your computer and film the billiard table, preferably from above, framing the billiard table as best as possible.

Your video projector must project on the billard board 

## How does it work ?

Before all, you must be sure that what is on the screen of your computer is also projected on the billard board. To do that, you must change your display setting to duplicate your screen on all other screen (in this case, a video projector).

There are four files numbered from 1 to 4:

**1.** The first file "1-Camera.py" will let you choose which camera you want to use to interact with the pool table.

**2.** The second file "2-Calibration.py" will ask you to mark the corners of the billiard table with a red dot, as in the picture:
![Step2](https://user-images.githubusercontent.com/82355033/136162943-6e1aa60d-3e5e-4ee9-aec9-d7741124032d.png)
Then the program will project a square pattern on the billiard table. You will have to superimpose the four dots on the four dots projected on the billiard table:
![Step22](https://user-images.githubusercontent.com/82355033/136165042-bf947717-2973-4c40-ace8-ab084fb3d31d.png)

**3.** The third file, "3-Detection", will ask you to remove all the elements from the billiard table.

**4.** The fourth file will project live information on the billiard table.

Enjoy :-)

![Step5](https://user-images.githubusercontent.com/82355033/136165592-53749120-44e6-4f7a-9b4d-f0c549a649ba.jpg)

## Debug Mode

If you didn't have the material to test, you may use the small "debug-mode", that will be using a video file to simulate the video input. To launch that mode, you just have to rename the video "NOdebug.mp4" to "debug.mp4". Then, you will be able tu use the files 2, 3 and 4 (the one become meaningless because this is not a camera input but a file input) 
