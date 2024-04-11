from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime
import openpyxl
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.ensemble import VotingClassifier
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,profile_identification_type,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city)

        return render(request, 'RUser/Register1.html')
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Profile_Identification_Status(request):
        expense = 0
        kg_price=0
        if request.method == "POST":

            prof_idno= request.POST.get('prof_idno')
            name= request.POST.get('name')
            screen_name= request.POST.get('screen_name')
            statuses_count= request.POST.get('statuses_count')
            followers_count= request.POST.get('followers_count')
            friends_count= request.POST.get('friends_count')
            created_at= request.POST.get('created_at')
            location= request.POST.get('location')
            default_profile= request.POST.get('default_profile')
            prf_image_url= request.POST.get('prf_image_url')
            prf_banner_url= request.POST.get('prf_banner_url')
            prf_bgimg_https= request.POST.get('prf_bgimg_https')
            prf_text_color= request.POST.get('prf_text_color')
            profile_image_url_https= request.POST.get('profile_image_url_https')
            prf_bg_title= request.POST.get('prf_bg_title')
            profile_background_image_url= request.POST.get('profile_background_image_url')
            description= request.POST.get('description')
            Prf_updated = request.POST.get('Prf_updated')

            df = pd.read_csv('Profile_Datasets.csv')

            def clean_text(text):
                '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
                and remove words containing numbers.'''
                text = text.lower()
                text = re.sub('\[.*?\]', '', text)
                text = re.sub('https?://\S+|www\.\S+', '', text)
                text = re.sub('<.*?>+', '', text)
                text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
                text = re.sub('\n', '', text)
                text = re.sub('\w*\d\w*', '', text)
                text = re.sub('"@', '', text)
                text = re.sub('@', '', text)
                text = re.sub('https: //', '', text)
                text = re.sub('Ã¢â‚¬â€', '', text)
                text = re.sub('\n\n', '', text)

                return text

            df['processed_content'] = df['name'].apply(lambda x: clean_text(x))

            def apply_results(label):
                if (label == 0):
                    return 0  # Fake
                elif (label == 1):
                    return 1  # Genuine

            df['results'] = df['Label'].apply(apply_results)

            cv = CountVectorizer(lowercase=False)

            y = df['results']
            X = df["id"].apply(str)

            print("X Values")
            print(X)
            print("Labels")
            print(y)

            X = cv.fit_transform(X)

            models = []
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            X_train.shape, X_test.shape, y_train.shape

            # SVM Model
            print("SVM")
            from sklearn import svm
            lin_clf = svm.LinearSVC()
            lin_clf.fit(X_train, y_train)
            predict_svm = lin_clf.predict(X_test)
            svm_acc = accuracy_score(y_test, predict_svm) * 100
            print(svm_acc)
            print("CLASSIFICATION REPORT")
            print(classification_report(y_test, predict_svm))
            print("CONFUSION MATRIX")
            print(confusion_matrix(y_test, predict_svm))
            models.append(('svm', lin_clf))

            print("KNeighborsClassifier")
            from sklearn.neighbors import KNeighborsClassifier
            kn = KNeighborsClassifier()
            kn.fit(X_train, y_train)
            knpredict = kn.predict(X_test)
            print("ACCURACY")
            print(accuracy_score(y_test, knpredict) * 100)
            print("CLASSIFICATION REPORT")
            print(classification_report(y_test, knpredict))
            print("CONFUSION MATRIX")
            print(confusion_matrix(y_test, knpredict))
            models.append(('KNeighborsClassifier', kn))

            classifier = VotingClassifier(models)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            prof_idno1 = [prof_idno]
            vector1 = cv.transform(prof_idno1).toarray()
            predict_text = classifier.predict(vector1)

            pred = str(predict_text).replace("[", "")
            pred1 = pred.replace("]", "")

            prediction = int(pred1)

            if prediction == 0:
                val = 'Fake Profile'
            elif prediction == 1:
                val = 'Genuine Profile'

            print(val)
            print(pred1)

            profile_identification_type.objects.create(
            prof_idno=prof_idno,
            name=name,
            screen_name=screen_name,
            statuses_count=statuses_count,
            followers_count=followers_count,
            friends_count=friends_count,
            created_at=created_at,
            location=location,
            default_profile=default_profile,
            prf_image_url=prf_image_url,
            prf_banner_url=prf_banner_url,
            prf_bgimg_https=prf_bgimg_https,
            prf_text_color=prf_text_color,
            profile_image_url_https=profile_image_url_https,
            prf_bg_title=prf_bg_title,
            profile_background_image_url=profile_background_image_url,
            description=description,
            Prf_updated=Prf_updated,
            Prediction=val)

            return render(request, 'RUser/Predict_Profile_Identification_Status.html',{'objs':val})
        return render(request, 'RUser/Predict_Profile_Identification_Status.html')

