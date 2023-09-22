from flask import Flask, render_template, request, session
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import tree

app = Flask(__name__)
app.secret_key = 'your_secret_key'

#List of the symptoms is listed here in list l1.

l1 = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain',
    'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue',
    'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy',
    'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating',
    'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite',
    'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
    'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes',
    'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
    'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements',
    'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity',
    'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
    'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech',
    'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness',
    'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell',
    'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
    'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body',
    'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
    'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
    'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
    'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum',
    'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads',
    'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
    'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
]


#List of Diseases is listed in list disease.

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
    'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
    ' Migraine','Cervical spondylosis',
    'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
    'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
    'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
    'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
    'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
    'Impetigo']

l2=[]
for i in range(0,len(l1)):
    l2.append(0)

#Reading the training .csv file
df=pd.read_csv("training.csv")
# Create the X_train and y_train variables.
X_train = df.drop("prognosis", axis=1)
y_train = df["prognosis"]
#Replace the values in the imported file by pandas by the inbuilt function replace in pandas.

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
    'Migraine':11,'Cervical spondylosis':12,
    'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
    'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
    'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
    'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
    '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
    'Impetigo':40}},inplace=True)

#printing the top 5 rows of the training dataset
df.head()


#Reading the  testing.csv file
tr=pd.read_csv("testing.csv")

#Using inbuilt function replace in pandas for replacing the values

tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
    'Migraine':11,'Cervical spondylosis':12,
    'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
    'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
    'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
    'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
    '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
    'Impetigo':40}},inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


#printing the top 5 rows of the testing data
tr.head()
doctors_df = pd.read_csv("doctors.csv")
remedies_df = pd.read_csv("symptom_precaution1.csv")
test_df = pd.read_csv("ttest.csv")
desc_df = pd.read_csv("symptom_Description.csv")

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('reg.html')

@app.route('/home', methods=['GET','POST'])
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/diet')
def diet():
    return render_template('diet.html')

@app.route('/Predict_disease')
def Predict_disease():
    return render_template('Predict_disease.html')

@app.route('/predict', methods=['GET', 'POST'])

def predict():
    if request.method == 'POST':
        # name = request.form['name']
        # if len(name) == 0:
        #     return render_template('index.html', message="Kindly fill the name.")
        symptom1 = request.form.get('symptom1')
        symptom2 = request.form.get('symptom2')
        symptom3 = request.form.get('symptom3')
        symptom4 = request.form.get('symptom4')
        symptom5 = request.form.get('symptom5')

        symptoms = [symptom1, symptom2, symptom3, symptom4, symptom5]
        print("Selected Symptoms : ",symptoms)          
        if len(symptoms) < 2:
            return render_template('index.html', error="Kindly fill at least the first two symptoms.")
        
        clf3 = tree.DecisionTreeClassifier()
        clf3.fit(X_train, y_train)

        inputtest = [0] * len(l1)
        for k in range(len(l1)):
            for z in symptoms:
                if z == l1[k]:
                    inputtest[k] = 1

        inputtest = [inputtest]

        predicted = clf3.predict(inputtest)[0]
        print("predicted : ",predicted)

        y_pred = clf3.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        cv_scores = cross_val_score(clf3, X_train, y_train, cv=5)
        mean_cv_accuracy = cv_scores.mean()
        print("Cross-validated accuracy:", mean_cv_accuracy)

        specialist = doctors_df.loc[doctors_df['Disease'] == predicted, 'Specialists'].values[0]
        print("Specialist:", specialist)
        session['specialist'] = specialist

        rem = remedies_df.loc[remedies_df['Disease'] == predicted, 'Combined_Precautions'].values[0]    
        print("precautions:", rem)

        test = test_df.loc[test_df['Disease'] == predicted, 'Diagnostic Tests'].values[0]
        print("test:", test)

        description = desc_df.loc[desc_df['Disease'] == predicted, 'Description'].values[0]
        print("desc:", description)

        # hospital = hospital_df.loc[hospital_df['District'] == specialist, 'Description'].values[0]
        # print("desc:", description)
        
        return render_template('Predict_disease.html', message= predicted,message2=rem, message3=test, message4=specialist, message5=description)
    

    return render_template('index.html')


@app.route('/hospital', methods=['POST'])

def hospital():
    if request.method == 'POST':
        hospital_df = pd.read_csv("newhospital.csv")
        district = request.form.get('district')
        print("district:", district)

        specialist = session.get('specialist')
        print(specialist)

        hospital = hospital_df.loc[(hospital_df['District'] == district) & (hospital_df['Specialist'] == specialist), 'Hospitals'].values[0]

        print("hospital:", hospital)
        
        return render_template('hospital.html', message6=hospital)   
if __name__ == '__main__':
    app.run(debug=True) 