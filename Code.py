import csv
with open('liver_dataset.csv','r') as csv_file:
    csv_reader= csv.reader(csv_file)

import pandas as pd
df=pd.read_csv('liver_dataset.csv')
# print(df)
liver_features=['Age','Gender','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin','Albumin_and_Globulin_Ratio']

#Extracting features
X=df.loc[:,liver_features].values
print(X)

#Extracting labels
Y=df.loc[:,['Dataset']].values

from sklearn import model_selection
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.2, random_state=0)

from sklearn import svm
#print(X_train)
#building an SVC using linear regression
#clf_ob=svm.SVC(kernel='linear',C=1).fit(X_train,Y_train)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)
print(clf.score(X_test,Y_test))

Y_train=Y_train.ravel()                              ### changes into 1-D array
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2
                            )
clf.fit(X_train, Y_train)
print(clf.feature_importances_)
print clf.score(X_test,Y_test)


####################PREDICTION################
predict= clf.predict([[23,0,2.3,0.8,509,28,44,6.9,2.9,0.7]])
print(predict)

import Tkinter
from Tkinter import *                  ### imports all Tkinter modules and functions


my=Tk()                                ### Forms main window called my
label1=Label(my,text='Age',justify=LEFT)
label2=Label(my,text='Gender (0 for female/ 1 for male)',justify=LEFT)
label3=Label(my,text='Total_Bilirubin (mg/dL)',justify=LEFT)
label4=Label(my,text='Direct_Bilirubin (mg/dL)',justify=LEFT)
label5=Label(my,text='Alkaline_Phosphotase (IU/L)',justify=LEFT)
label6=Label(my,text='Alamine_Aminotransferase (IU/L)',justify=LEFT)
label7=Label(my,text='Aspartate_Aminotransferase (IU/L)',justify=LEFT)
label8=Label(my,text='Total_Protiens (g/dL)',justify=LEFT)
label9=Label(my,text='Albumin (g/dL)',justify=LEFT)
label10=Label(my,text='Albumin_and_Globulin_Ratio',justify=LEFT)

label1.grid(row=0,column=1)            ### Setting Location of label1
label2.grid(row=1,column=1)
label3.grid(row=2,column=1)
label4.grid(row=3,column=1)
label5.grid(row=4,column=1)
label6.grid(row=5,column=1)
label7.grid(row=6,column=1)
label8.grid(row=7,column=1)
label9.grid(row=8,column=1)
label10.grid(row=9,column=1)

entry1=Entry(my)               ### Window to input values from patients
entry2=Entry(my)
entry3=Entry(my)
entry4=Entry(my)
entry5=Entry(my)
entry6=Entry(my)
entry7=Entry(my)
entry8=Entry(my)
entry9=Entry(my)
entry10=Entry(my)

entry1.grid(row=0,column=2)
entry2.grid(row=1,column=2)
entry3.grid(row=2,column=2)
entry4.grid(row=3,column=2)
entry5.grid(row=4,column=2)
entry6.grid(row=5,column=2)
entry7.grid(row=6,column=2)
entry8.grid(row=7,column=2)
entry9.grid(row=8,column=2)
entry10.grid(row=9,column=2)

import tkMessageBox
def CallBack():
   a1 = entry1.get()
   a2 = entry2.get()
   a3 = entry3.get()
   a4 = entry4.get()
   a5 = entry5.get()
   a6 = entry6.get()
   a7 = entry7.get()
   a8 = entry8.get()
   a9 = entry9.get()
   a10= entry10.get()
   prediction=clf.predict([[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]])
   if prediction ==1:
     tkMessageBox.showinfo( 'Result',"You may have a liver related problem and "
                          "are thus advised to consult a doctor")
   if prediction ==2:
     tkMessageBox.showinfo('Result',"You may not have a liver related problem "
                          "but it is advised you regularly have your health checkup done" )
B = Tkinter.Button(my, text ="Ok", command = CallBack)
B.grid(row=10, column=2)
my.mainloop()              ### means the box/window will be packed and get displayed

###################VISUALISING#################

import matplotlib.pyplot as plt
B = df['Dataset']          # Split off classifications
A = df.ix[:, :'Dataset'] # Split off features

# Two different scatter series so the class labels in the legend are distinct
plt.scatter(A[B==1]['Age'], A[B==1]['Direct_Bilirubin'], label='Class 1', c='red')
plt.scatter(A[B==2]['Age'], A[B==2]['Direct_Bilirubin'], label='Class 2', c='blue')
# Prettify the graph
plt.legend()
plt.xlabel('Age')
plt.ylabel('Direct_Bilirubin')

# display
plt.show()

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
standardscalar = StandardScaler()
x_std=standardscalar.fit_transform(X_train)
# print(x_std)
import numpy as np

from sklearn.manifold import TSNE
tsne=TSNE(n_components=2, random_state=0) #n_components means how many dimensions we want the end result to be in
x_test_2d=tsne.fit_transform(x_std)
# print (x_test_2d)

list=x_test_2d.tolist()
print len(df)
print Y_train
for i,b in zip(range(195),Y_train):
        if b==1:
            c='r'
        elif b==2:
            c='g'
        plt.scatter(list[i][0],list[i][1],marker='o',c=c)
dot1=plt.scatter(list[0][0],list[0][1],marker='o',c='r')
dot2=plt.scatter(list[8][0],list[8][1],marker='o',c='g')
plt.legend((dot1,dot2),('Patients with liver disease','Patients without liver disease'))
plt.xlabel('x-tsne')
plt.ylabel('y-tsne')
plt.show()













