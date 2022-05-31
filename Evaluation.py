import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix

def Evaluation(y_test,y_pred):
    com=confusion_matrix(y_test,y_pred)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
                    com.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in
              zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2, 2)
    ax = sns.heatmap(com, annot=labels, fmt='', cmap='Blues')
    ax.set_title('Seaborn Confusion Matrix with labels');
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ');
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    print(f"Accuracy : {metrics.accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Recall positive : {metrics.recall_score(y_test,y_pred,pos_label=1)*100:.2f}%")
    print(f"Recall negative : {metrics.recall_score(y_test,y_pred,pos_label=0)*100:.2f}%")
    print(f"precision postive : {metrics.precision_score(y_test,y_pred,pos_label=1)*100:.2f}%")
    print(f"precision negative : {metrics.precision_score(y_test,y_pred,pos_label=0)*100:.2f}%")
    print(f"The Mean Squared Error : {metrics.mean_squared_error(y_test, y_pred) * 100:.2f}%")
    plt.show()

def voting ( model_1,model_2,model_3,model_4,model_5):
    result = model_1 + model_2 + model_3 + model_4 + model_5
    l=len(result)
    for x in range(0,l):
        y=result[x]
        if y >= 3:
            result[x]=1
        else:
            result[x]=0
    return result