import os
import cv2
from PIL import Image
import climage 

# converts the image to print in terminal 
# inform of ANSI Escape codes 

# from PIL import Image                                                                                
# Clearing the Screen

def linear():
    ## For Linear Data Structure 
    while True:
        print ("Enter 1 for Array Image Data Structure :")
        print ("Enter 2 for Linked List Data Structure:  ")
        print ("Enter 0 for main menu : ")
        ask = input("")
        if ask=="1":
            output = climage.convert('array.png') 

            # prints output on console. 
            print(output)

#             img = Image.open('array.png')
#             img.show()
# 
#             img = cv2.imread("array.png")
#             img = Image.fromarray(img, "RGB")
#             img.show()
        elif ask == "2":
            pass
        elif ask == "0":
            os.system('cls')
            break
        else:
            print ("Invalid Input")
count = 0
while True:
    if count == 0:
        print ("This is a Learning Program of Data Structures and Algorithms")
        count += 1
    print ("Enter 1 for Linear Data Structure :")
    print ("Enter 2 for Non - Linear Data Structure:  ")
    print ("Enter 0 for exit : ")
    ask = input("")
    if ask=="1":
        linear()
    elif ask == "2":
        pass
    elif ask == "0":
        break
    else:
        print ("Invalid Input")
        