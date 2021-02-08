# Object Character Recognition on an Invoice

## What it does:
This is a simple script that extracts text from an invoice using OpenCV and PyTesseract. Enter the name of the person whose invoice you want to extract and the script will detect and print the following elements:

1) Customer Name
2) Address 
3) Phone Number
4) Date of Transaction
5) Description of Good
6) Unit Price
7) Amount
8) Sub Total 
9) Tax Rate
10) Tax Amount
11) Total Amount to be Paid 

## Process: 
Firstly, the script consideres a query image of a blank invoice which it stores for later use. It then stores the keypoints and descriptors of that image using 5000 orbs created using the cv2.ORB_create() method. Secondly, it initializes a list which stores the pixel locations of all the features to be detected along with their names. One can use Microsoft Paint to manually locate the points or use a script that returns locations of all points, the link for which is attached below. Thirdly, a function is definded that inputs an image name, displays it raw and stores the keypoints and descriptors using the orb.detectAndCompute() method. It then feeds the descriptors into a Brute Force matcher to detect the matching features between the query image and the input image. After calculating the source and destination points from 25% of the best matches, it uses cv2.findHomography() and cv2.warpPerspective() methods to align the input image the same way as the query image, and displays it. In layman’s terms, if the input image is curved or rotated, this function will straighten it, somewhat like CamScanner, Adobe Scan, etc. Fourthly and finally, the script crops the initialized features from the newly aligned input image which are finally fed into PyTesseract which detects the text found in every cropped feature. All the detections are printed in the console and cv2.putText() is used to display the detections on top of the input image for reference. 

## Output Gif:
Here's a gif showing the script in action
![Alt text](InvoiceOCR.gif) 

## Pros: 
1) Quick and Easy - Just input the image/person name and the features are detected in less than twenty seconds.
2) Deployable on large datasets - A quickly defined function can loop the script over many invoices in a dataset and store the same in a .csv file. 
3) Flexible - Works even if the input image is unaligned/cropped/rotated as long as the text is readable

## Cons:
1) Limited templates - Only works on one invoice template for now, linked below. Other templates must have feature locations extracted seperately.
2) Doesn't work well on Colab - Due to cv2.imshow() method not being supported. Images have to be printed like plots instead of being shown in seperate windows. 

### Click here to view the Colab version of this script:
https://colab.research.google.com/drive/18Sti-KwZypNo3bg8WRX7ElUalZqyPhPy

#### References: 
1) PyTesseract: https://github.com/tesseract-ocr/tesseract
2) OpenCV: https://docs.opencv.org/master/index.html
3) Query Image: http://www.work-template.com/printable-blank-invoice-template.html
4) Learn PyTessseract for document extraction: https://www.murtazahassan.com/courses/opencv-projects
