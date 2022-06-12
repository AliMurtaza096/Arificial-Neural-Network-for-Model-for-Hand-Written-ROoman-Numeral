
import os 

import cv2
import matplotlib.image as img
import  numpy as np


from sklearn.neural_network import MLPClassifier



current_dir = os.getcwd()
paths = []
for dirpath,dirname,filenames in os.walk(current_dir):
    paths.append(dirpath)
train_path =  paths[1]
trainDir_paths = paths[2:12]

# print(train_path,trainDir_paths)
val_path = paths[12]
valDir_paths = paths[13:]



images = []
indexes_of_classes = []
for i in trainDir_paths: 
    os.chdir(i)
    count = 0
    for j in os.scandir(i):
        count +=1
        img_name = j.name
        
        image = cv2.imread(img_name,0)
        images.append(image)
    indexes_of_classes.append(count)


X_train = []




for image in range(len(images)):
    # print(images[image].shape)
    # temp = np.array([])
    img = images[image]
    
    # print(img.shape)
    # print(img[70])
    row,col = img.shape[0],img.shape[1]
    row,col = row // 2,col // 2
    


    # forth_prev= img[row-4][col-60:col+60]
    third_prev= img[row-3][col-60:col+60]
    second_prev= img[row-2][col-60:col+60]
    prev_center = img[row-1][col-60:col+60]
    center_array = img[row][col-60:col+60]
    next_center = img[row+1][col-60:col+60]
    sec_next_center = img[row+2][col-60:col+60]
    center_array = third_prev+second_prev+prev_center+center_array+next_center+sec_next_center
  
#

X_train = np.array(X_train)

y_train = []

labels = [1,2,3,4,9,5,6,7,8,10]
for index in range(len(indexes_of_classes)):
    for j in range(indexes_of_classes[index]):
        # print(index)  
        y_train.append(labels[index])





test_images = []
test_indexes = []
for i in valDir_paths: 
    os.chdir(i)
    count = 0
    for j in os.scandir(i):
        count +=1
        img_name = j.name
        
        image = cv2.imread(img_name,0)
        test_images.append(image)
    test_indexes.append(count)


X_test = []

for image in range(len(test_images)):
    # print(images[image].shape)
    img = test_images[image]
    
    row,col = img.shape[0],img.shape[1]
    row,col = row // 2,col // 2
   

    # forth_prev= img[row-4][col-60:col+60]
    third_prev= img[row-3][col-60:col+60]
    second_prev= img[row-2][col-60:col+60]
    prev_center = img[row-1][col-60:col+60]
    center_array = img[row][col-60:col+60]
    next_center = img[row+1][col-60:col+60]
    sec_next_center = img[row+2][col-60:col+60]
    center_array =third_prev+second_prev+prev_center+center_array+next_center+sec_next_center

    X_test.append(center_array)


y_test = []

labels = [1,2,3,4,9,5,6,7,8,10]
for index in range(len(test_indexes)):
    for j in range(test_indexes[index]):
        # print(index)  
        y_test.append(labels[index])




# print(y_train)
clf = MLPClassifier(hidden_layer_sizes = (750,750),random_state=None, max_iter=400,solver='adam',activation = 'logistic',shuffle=True,learning_rate_init=0.0001,verbose=True).fit(X_train,y_train)
# print(clf)
predict =clf.predict(X_test)


a = clf.score(X_test,y_test)    
print(a,'SCORE')  

# print(len(X_train))



    
