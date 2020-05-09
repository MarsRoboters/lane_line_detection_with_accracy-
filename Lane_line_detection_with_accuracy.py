#!/usr/bin/env python
# coding: utf-8

# In[19]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[20]:


def canny(image):
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


# In[21]:


def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
                x1,y1,x2,y2 = line.reshape(4)
                cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
    return line_image


# In[22]:


def region_of_interest(image):
    height = image.shape[0]
    width  = image.shape[1]
    polygons = np.array([[(100, height),(1100, height), (600, 400)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


# In[23]:


image = cv2.imread('test_image.png')
plt.imshow(image, cmap = 'gray')


# In[24]:


lane_image = np.copy(image)


# In[25]:


canny_image = canny(lane_image)
plt.imshow(canny_image, cmap = 'gray')


# In[26]:


cropped_image = region_of_interest(canny_image)
plt.imshow(cropped_image, cmap = 'gray')


# In[27]:


cropped_image_acc_gray = np.copy(cropped_image)
# line_image_ = cv2.cvtColor(cropped_image_acc, cv2.COLOR_RGB2GRAY)
cropped_image_acc_gray[cropped_image_acc_gray[:]<10] = 0
cropped_image_acc_gray[cropped_image_acc_gray[:]>10] = 255

plt.imshow(cropped_image_acc_gray, cmap = 'gray')


# In[28]:


lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)


# In[29]:


line_image = display_lines(lane_image, lines)
plt.imshow(line_image, cmap = 'gray')


# In[30]:


line_image_acc = np.copy(line_image)
line_image_acc_gray = cv2.cvtColor(line_image_acc, cv2.COLOR_RGB2GRAY)
line_image_acc_gray[line_image_acc_gray[:]<10] = 0
line_image_acc_gray[line_image_acc_gray[:]>10] = 255

plt.imshow(line_image_acc_gray, cmap = 'gray')


# In[31]:


o_img = np.asarray(cropped_image_acc_gray)
p_img = np.asarray(line_image_acc_gray)


# In[32]:


dist = np.linalg.norm(o_img - p_img)


# In[33]:


Accuracy_percentage = 100 - dist/100

Accuracy_percentage


# In[34]:


combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
plt.imshow(combo_image, cmap = 'gray')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




