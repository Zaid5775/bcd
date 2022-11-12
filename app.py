from flask import Flask , render_template , request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


app = Flask(__name__,
            static_url_path='',
           
            template_folder='template')


@app.route('/')
def home(): 
          return render_template('home.html')
      
      





@app.route('/predict', methods=["GET","POST"])

def predict(): 
    
   
    pmean = int(request.form['pmean']);
    amean =  int(request.form['amean']);
    wmean = int(request.form['wmean']);
    awmean = int(request.form['awmean']);
    pwmean = int(request.form['pwmean'])
    rmean = int(request.form['rmean'])
    cmean = int(request.form['cmean']);
    
    breast_cancer_data = pd.read_csv ('data.csv')
    breast_corr = breast_cancer_data.corr ()
    corr_value = breast_corr.iloc [0, :].values
    new_corr_value_list = []
    col_index = []

    for i in range (len (breast_cancer_data.columns)):
        if (corr_value[i] > 0.7):
            new_corr_value_list.append (corr_value[i])
        if i == 0:
            continue
        col_index.append (i)
    
    new_corr_value_list.remove (1.0)
    features_data = breast_cancer_data.iloc [:, col_index]
    X = breast_cancer_data.iloc [:, col_index].values
    target = breast_cancer_data.iloc [:, 0]
    y = breast_cancer_data.iloc [:, 0].values 
    x_train, x_test, y_train, y_test = train_test_split (X, y, test_size=0.3, random_state=42)
    LR = LogisticRegression ()
    LR.fit (x_train, y_train)
    
    #cancer = ""
    def predict_cancer (lst):
        
        if LR.predict (lst)[0] == 0:
            #cancer = "Benign tumors are those that stay in their primary location without invading other sites of the body. They do not spread to local structures or to distant parts of the body. Benign tumors tend to grow slowly and have distinct borders/n Benign tumors are not usually problematic. However, they can become large and compress structures nearby, causing pain or other medical complications. For example, a large benign lung tumor could compress the trachea (windpipe) and cause difficulty in breathing. This would warrant urgent surgical removal. Benign tumors are unlikely to recur once removed. Common examples of benign tumors are fibroids in the uterus and lipomas in the skin."
            return ('Benign Cancer detected !!!/n', cancer)
        else:
            #cancer = "Benign tumors are those that stay in their primary location without invading other sites of the body. They do not spread to local structures or to distant parts of the body. Benign tumors tend to grow slowly and have distinct borders/n Benign tumors are not usually problematic. However, they can become large and compress structures nearby, causing pain or other medical complications. For example, a large benign lung tumor could compress the trachea (windpipe) and cause difficulty in breathing. This would warrant urgent surgical removal. Benign tumors are unlikely to recur once removed. Common examples of benign tumors are fibroids in the uterus and lipomas in the skin."
            return ('Malignant Cancer detected !!! /n' , cancer)

    patient_detail = np.array ([], dtype='int64')
    
    l1 =  [[pmean,amean, wmean , awmean ,pwmean, rmean,cmean,0.2419,0.2419	,0.07871,	1.095	,0.9053,	8.589	,153.4,	0.006399,	0.04904,	0.05373,	0.01587,	0.03003	,0.006193	,25.38	,17.33	,184.6	,2019	,0.1622	,0.6656	,0.7119,	0.2654,	0.4601	,0.1189]]

    patient_detail = np.append (patient_detail, l1)
    result = predict_cancer (patient_detail.reshape (1, -1))    

    
    
    
    
    return render_template('home.html' , pred = result )  
               

    
@app.route('/clear') 
def clear():
    if(render_template('predict')):
        render_template('home.html')
    
           

if __name__ == '__main__':
    	app.run(debug=True)
 