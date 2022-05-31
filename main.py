import os
import customtkinter as customtkinter
import pandas as pd
from Models import *
from Evaluation import *
from Data_Cleansing import *
from tkinter import filedialog
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest
from tkinter import messagebox
from PIL import Image, ImageTk

#-------load dataset-------

path=r'C:\Users\hesha\PycharmProjects\Tumor Cancer Prediction_Data.csv'
cancer_data = pd.read_csv(path)

#-------Data_Cleansing-------

print(cancer_data.shape)
print(cancer_data.head(10))
dataCleansing = Data_Cleansing()
cancer_data= cancer_data.drop('Index', axis=1)
cancer_data.dropna(inplace=True)
print("-------Number of NaN in every column .-------")
print (cancer_data.isnull().sum())
cancer_data.drop_duplicates(inplace=True)
cancer_data = dataCleansing.Encode(cancer_data)
# labelEncoder_Y = LabelEncoder()
# cancer_data['diagnosis']=labelEncoder_Y.fit_transform(cancer_data['diagnosis'].values)
cancer_data = dataCleansing.Remove_outlier(cancer_data)
cancer_data=dataCleansing.Handling_Zeros(cancer_data)
cancer_data = dataCleansing.normalization(cancer_data)
print(cancer_data.shape)

#----featcure selection-----

plt.figure(figsize=(15,8))
cancer_data.corr()['diagnosis'].sort_values(ascending = False).plot(kind='bar')
plt.show()
columns = cancer_data.iloc[:,0:30]
target = cancer_data['diagnosis']
bestfeatures = SelectKBest(score_func=chi2, k=10)

#--------Split_Data---------

column = bestfeatures.fit_transform(columns,target)
print(column.shape)
X_train, X_test, y_train, y_test = train_test_split(column, target, test_size=0.250,random_state=0)

#-------Classification-------

print("Logistic Regression model: ")
logreg1 = Logistic(X_train, y_train, X_test, y_test)
logreg1.save_model()
logistic_model = pickle.load(open('logistic_model', 'rb'))
y_pred_logreg = logistic_model.predict(X_test)
Evaluation(y_test,y_pred_logreg)
print('_____________________________________________')

print("Decision Tree model: ")
dtree=DecisionTree(X_train, y_train, X_test, y_test)
dtree.save_model()
Dtree_model = pickle.load(open('dtree_model', 'rb'))
y_pred_Dtree = Dtree_model.predict(X_test)
Evaluation(y_test,y_pred_Dtree)
print('_____________________________________________')

print("SVC method of svm class to use Support Vector model: ")
svc=Svc_kernal(X_train, y_train, X_test, y_test)
svc.save_model()
svc_model = pickle.load(open('svc_model', 'rb'))
y_pred_svc = svc_model.predict(X_test)
Evaluation(y_test,y_pred_svc)
print('_____________________________________________')

print("Random Forest model: ")
Rforest = Random_forest(X_train, y_train, X_test, y_test)
Rforest.save_model()
random_model = pickle.load(open('random_model', 'rb'))
y_pred_random = random_model.predict(X_test)
Evaluation(y_test,y_pred_random)
print('_____________________________________________')

print("svc polynomial model: ")
svc_polynomial=Svc_Polynomial(X_train, y_train, X_test, y_test)
svc_polynomial.save_model()
polynomial_model = pickle.load(open('polynomial_model', 'rb'))
y_pred_poly= polynomial_model.predict(X_test)
Evaluation(y_test,y_pred_poly)
print('_____________________________________________')

ls = []
def openFile():
    filepath = filedialog.askopenfilename(initialdir="", title="choose tha test file ",filetypes=[("CSV files", '.csv'), ('Text Docs', '.txt'), ('All types', '.*')])
    lineOne = f"Uploaded successfully"
    allLines = [lineOne]
    messagebox.showinfo("Successfully", "\n".join(allLines))
    if (filepath):
        global count
        dataCleaning = Data_Cleansing()
        count = 0
        ls.clear()
        label_info_2.set_text(filepath)
        test_data = pd.read_csv(filepath)
        test_data = test_data.drop('Index', axis=1)
        test_data.dropna(inplace=True)
        test_data.drop_duplicates(inplace=True)
        # test_data = dataCleaning.Encode(cancer_data)
        # test_data = dataCleaning.Remove_outlier(cancer_data)
        test_data = dataCleaning.Handling_Zeros(test_data)
        test_data = dataCleaning.normalization(test_data)
        X = bestfeatures.transform(test_data)
        y_logreg = logistic_model.predict(X)
        y_svc = svc_model.predict(X)
        y_dtree = Dtree_model.predict(X)
        y_poly = polynomial_model.predict(X)
        y_random = random_model.predict(X)
        print(y_logreg)
        print(y_svc)
        print(y_dtree)
        print(y_poly)
        print(y_random)
        pred_vote = voting(y_svc, y_poly, y_dtree, y_logreg, y_random)
        l = len(pred_vote)
        for x in range(0 , l):
            y = pred_vote[x]
            if y == 0:
                ls.append(" Fortunately , It is Benign cells ")
            else:
                ls.append(" Unfortunately , It is Malignant cells ")
count = 0
def nextState():
    global count
    if count < len(ls):
        label_info_1.set_text(f"{ls[count]} \n" + f"{count+1}")
        count = count + 1
    else :
        messagebox.showerror("Error" , "Data Finshed.")


PATH = os.path.dirname(os.path.realpath(__file__))

customtkinter.set_appearance_mode("system")
customtkinter.set_default_color_theme("blue")

root_tk = customtkinter.CTk()
root_tk.geometry("400x400")
root_tk.title("Cancer Prediction")


def change_mode():
    if switch_2.get() == 1:
        customtkinter.set_appearance_mode("dark")
    else:
        customtkinter.set_appearance_mode("light")
image_size = 20
add_folder_image = ImageTk.PhotoImage(
    Image.open(PATH + "/test_images/add-folder.png").resize((image_size, image_size), Image.ANTIALIAS))
pred_image = ImageTk.PhotoImage(
    Image.open(PATH + "/test_images/analytics.png").resize((image_size, image_size), Image.ANTIALIAS))
exit_image = ImageTk.PhotoImage(
    Image.open(PATH + "/test_images/exit.png").resize((image_size, image_size), Image.ANTIALIAS))

root_tk.grid_rowconfigure(0, weight=1)
root_tk.grid_columnconfigure(0, weight=1, minsize=200)

frame_1 = customtkinter.CTkFrame(master=root_tk, width=250, height=240, corner_radius=15)
frame_1.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

label_info_2 = customtkinter.CTkLabel(master=root_tk , text = "Path Here" , fg_color=("white","grey38") , text_font=("Arial" , 7))

label_info_2.grid(row=2, column=0, columnspan=1, pady=(0,10),padx=5, sticky="ew")

frame_1.grid_columnconfigure(0, weight=1)
frame_1.grid_columnconfigure(1, weight=1)
frame_1.grid_rowconfigure(0, minsize=10)

label_info_1 = customtkinter.CTkLabel(master=frame_1,
                                      text="State \n" + "0",
                                      height=50,
                                      fg_color=("white", "gray38"))

label_info_1.grid(row=4, column=0, columnspan=3, padx=20, pady=10, sticky="ew")

button_1 = customtkinter.CTkButton(master=frame_1, image=add_folder_image, text="Add File", width=190, height=40,
                                   compound="right", command=openFile)
button_1.grid(row=1, column=0, columnspan=2, padx=20, pady=10, sticky="ew")

predBtn = customtkinter.CTkButton(master=frame_1, image=pred_image, text="Next State", width=190, height=40,
                                  compound="right", fg_color="#14C38E", hover_color="#36AE7C",
                                  command=nextState)
predBtn.grid(row=2, column=0, columnspan=2, padx=20, pady=10, sticky="ew")

switch_2 = customtkinter.CTkSwitch(master=frame_1,
                                   text="Dark Mode",
                                   command=change_mode)
switch_2.grid(row=6, column=0, columnspan=2, padx=20, pady=20, sticky="ew")

exit = customtkinter.CTkButton(master=frame_1, image=exit_image, text="Exit", width=190, height=40,
                               compound="right", fg_color="#DD4A48", hover_color="#D35B58",
                               command=root_tk.destroy)
exit.grid(row=7, column=0, columnspan=2, padx=20, pady=10, sticky="ew")

root_tk.mainloop()