# Interactive Billard (Also know as Pool Table)
 
The Interative Billard will project useful informations (such as speed, acceleration...) directly on the board.

https://user-images.githubusercontent.com/82355033/136162758-bdefb44f-0264-434b-a80b-5c885dc3d04b.mp4

This interactive billiards aims to explore how new pedagogical platforms can be designed thanks to tangible media and artificial intelligence. Neuroscience research raise the importance of live experiences in cognitive process in particular during the learning process. In this game, players discover and learn physic and math concepts according to their progress in the pool game. Based on artificial intelligence technics, this augmented billiards provides guidance and information such as dynamic equation resolutions or geometric projections to the players thanks to a video-projector. We hope that such kind of learning platforms will be broadly used in future education systems.

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

**Nota Bene**: Running the three first files is only for the initialisation ; as long as you didn't change your setting conditions (camera position, video projector position or luminosity condition), you just need to run the last file "4-SimpleBillard.py"

## Debug Mode

If you didn't have the material to test, you may use the small "debug-mode", that will be using a video file to simulate the video input. To launch that mode, you just have to rename the video "NOdebug.mp4" to "debug.mp4". Then, you will be able tu use the files 2, 3 and 4 (the one become meaningless because this is not a camera input but a file input)

## Customization

### Background

You can customize the background by simply change the image "FondDVIC.png" for another.

## Save Parameters

There is 3 importants paramters that are save:

+ The information about what camera to use is save in *camera.json*
+ The information about the different perspective deformations for the camera and the video projector are store in the file *data.json*
+ The image of the table board clear of balls is save the image file *background.jpg*
