import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import numpy as np
df=pd.read_csv("healthcare_dataset.csv")
#to describe mean,median and other statistics data
a=df.describe()
print(a)

#to plot histogram
for i in df.select_dtypes(include="number"):
    sns.histplot(data=df,x=i)
    plt.show()
#To find correlation matrix
s=df.select_dtypes(include="number").corr()
sns.heatmap(s,annot=True)
inf=df.info()
print(inf)
#To Identify patterns and trends
for i in ['Room Number ','Age']:
    plt.scatter(data=df,x=i,y=' Billing Amount')
    plt.show()
#to identify anomolies
for i in df.select_dtypes(include="number"):
    sns.boxplot(data=df,x=i)
    plt.show()
#make basic feature lever level inference using visuals

img = cv2.imread('krishna.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


mean_val = np.mean(gray)
std_dev = np.std(gray)
min_val, max_val, _, _ = cv2.minMaxLoc(gray)

print(f"Mean Pixel Intensity: {mean_val}")
print(f"Standard Deviation: {std_dev}")
print(f"Min Pixel Value: {min_val}, Max Pixel Value: {max_val}")
edges = cv2.Canny(gray, threshold1=100, threshold2=200)


plt.imshow(edges, cmap='gray')
plt.title("Edge Detection (Canny)")
plt.axis("off")
plt.show()


