# Object Character Recognition on an Invoice

This is a simple script that extracts text from an invoice using OpenCV and PyTesseract. Enter the name of the person whose invoice you want to extract and the script will detect and print elements such as:

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

Firstly, the script consideres a query image of a blank invoice which it stores for later use. It then stores the keypoints and descriptors of that image using 5000 orbs created using the cv2.ORB_create() method. Secondly, it initializes a list which stores the pixel locations of all the features to be detected along with their names. One can use Microsoft Paint to manually locate the points or use a script that returns locations of all points, the link for which is attached below. Thirdly, a function is definded that inputs the image name, displays it as it is and stores the keypoints and descriptors of th

![Alt text](InvoiceOCR.gif) 

## Click here to view the Colab version of this script:
https://colab.research.google.com/drive/18Sti-KwZypNo3bg8WRX7ElUalZqyPhPy

### References: 
1) PyTesseract: https://github.com/tesseract-ocr/tesseract
2) OpenCV: https://docs.opencv.org/master/index.html
3) Query Image: http://www.work-template.com/printable-blank-invoice-template.html
4) Learn PyTessseract for document extraction: https://www.murtazahassan.com/courses/opencv-projects
