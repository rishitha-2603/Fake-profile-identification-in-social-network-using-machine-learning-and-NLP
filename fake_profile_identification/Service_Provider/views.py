from django.db.models import  Count, Avg
from django.shortcuts import render, redirect
from django.db.models import Count
from django.db.models import Q
import datetime
import xlwt
from django.http import HttpResponse

import string
import re
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
import openpyxl

# Create your views here.
from Remote_User.models import ClientRegister_Model,profile_identification_type,detection_ratio,detection_accuracy


def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password =="Admin":
            return redirect('View_Remote_Users')

    return render(request,'SProvider/serviceproviderlogin.html')



def View_Profile_Identity_Prediction(request):

    obj = profile_identification_type.objects.all()
    return render(request, 'SProvider/View_Profile_Identity_Prediction.html', {'objs': obj})

def View_Profile_Identity_Prediction_Ratio(request):
    detection_ratio.objects.all().delete()
    ratio = ""
    kword = 'Genuine Profile'
    print(kword)
    obj = profile_identification_type.objects.all().filter(Prediction=kword)
    obj1 = profile_identification_type.objects.all()
    count = obj.count();
    count1 = obj1.count();
    ratio = (count / count1) * 100
    if ratio != 0:
        detection_ratio.objects.create(names=kword, ratio=ratio)

    ratio1 = ""
    kword1 = 'Fake Profile'
    print(kword1)
    obj1 = profile_identification_type.objects.all().filter(Prediction=kword1)
    obj11 = profile_identification_type.objects.all()
    count1 = obj1.count();
    count11 = obj11.count();
    ratio1 = (count1 / count11) * 100
    if ratio1 != 0:
        detection_ratio.objects.create(names=kword1, ratio=ratio1)


    obj = detection_ratio.objects.all()
    return render(request, 'SProvider/View_Profile_Identity_Prediction_Ratio.html', {'objs': obj})


def View_Remote_Users(request):
    obj=ClientRegister_Model.objects.all()
    return render(request,'SProvider/View_Remote_Users.html',{'objects':obj})


def charts(request,chart_type):
    chart1 = detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def charts1(request,chart_type):
    chart1 = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts1.html", {'form':chart1, 'chart_type':chart_type})

def likeschart(request,like_chart):
    charts =detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})

def likeschart1(request,like_chart):
    charts =detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart1.html", {'form':charts, 'like_chart':like_chart})

def Download_Trained_DataSets(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="Predicted_Datasets.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = profile_identification_type.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:
        row_num = row_num + 1

        ws.write(row_num, 0, my_row.prof_idno, font_style)
        ws.write(row_num, 1, my_row.name, font_style)
        ws.write(row_num, 2, my_row.screen_name, font_style)
        ws.write(row_num, 3, my_row.statuses_count, font_style)
        ws.write(row_num, 4, my_row.followers_count, font_style)
        ws.write(row_num, 5, my_row.friends_count, font_style)
        ws.write(row_num, 6, my_row.created_at, font_style)
        ws.write(row_num, 7, my_row.location, font_style)
        ws.write(row_num, 8, my_row.default_profile, font_style)
        ws.write(row_num, 9, my_row.prf_image_url, font_style)
        ws.write(row_num, 10, my_row.prf_banner_url, font_style)
        ws.write(row_num, 11, my_row.prf_bgimg_https, font_style)
        ws.write(row_num, 12, my_row.prf_text_color, font_style)
        ws.write(row_num, 13, my_row.profile_image_url_https, font_style)
        ws.write(row_num, 14, my_row.prf_bg_title, font_style)
        ws.write(row_num, 15, my_row.profile_background_image_url, font_style)
        ws.write(row_num, 16, my_row.description, font_style)
        ws.write(row_num, 17, my_row.Prf_updated, font_style)
        ws.write(row_num, 18, my_row.Prediction, font_style)


    wb.save(response)
    return response

def Train_Test_DataSets(request):

    detection_accuracy.objects.all().delete()

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
            return 0 # Fake
        elif (label == 1):
            return 1 # Genuine

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
    print("X_test")
    print(X_test)
    print(X_train)

    print("Naive Bayes")

    from sklearn.naive_bayes import MultinomialNB
    NB = MultinomialNB()
    NB.fit(X_train, y_train)
    predict_nb = NB.predict(X_test)
    naivebayes = accuracy_score(y_test, predict_nb) * 100
    print("ACCURACY")
    print(naivebayes)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, predict_nb))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, predict_nb))
    detection_accuracy.objects.create(names="Naive Bayes", ratio=naivebayes)


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
    detection_accuracy.objects.create(names="SVM", ratio=svm_acc)


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
    detection_accuracy.objects.create(names="KNeighborsClassifier", ratio=accuracy_score(y_test, knpredict) * 100)

    obj = detection_accuracy.objects.all()

    return render(request,'SProvider/Train_Test_DataSets.html', {'objs': obj})














