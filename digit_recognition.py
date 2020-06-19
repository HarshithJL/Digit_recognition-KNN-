import numpy as np
import cv2

img  =  cv2.imread("train.png",cv2.IMREAD_GRAYSCALE)
test = cv2.imread("test.png",cv2.IMREAD_GRAYSCALE)

rows = np.vsplit(img,50)
cells = []

# cv2.imshow("image",img)
# cv2.waitKey(0)
for row in rows:
    row_cells = np.hsplit(row,50)
    for rowCells in row_cells:
        rowCells = rowCells.flatten()
        cells.append(rowCells)

cells = np.array(cells,dtype=np.float32)#converting normal array to numpy array for faster computation
k = np.arange(10)
cell_labels = np.repeat(k,250)


test_inputs = np.vsplit(test,50)#test image and train image size should be same
test_digits = []
cv2.imshow("input",test_inputs[28])
cv2.waitKey(0)
test_digits.append(test_inputs[28].flatten())
# for d in test_inputs:
#     d = d.flatten()
#     test_digits.append(d)
#
test_digits = np.array(test_digits,dtype=np.float32)

#knn setup
knn = cv2.ml.KNearest_create()
knn.train(cells,cv2.ml.ROW_SAMPLE,cell_labels)

ret,result,neighbours,dist = knn.findNearest(test_digits,k=1)

print("This is",result[0][0])
