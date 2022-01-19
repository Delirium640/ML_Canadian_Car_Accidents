import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('clf_lgb.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]

    c_year = int(request.form.get('c_year'))
    c_mnth = int(request.form.get('c_mnth'))
    c_wday = int(request.form.get('c_wday'))
    c_hour = int(request.form.get('c_hour'))

    c_conf = int(request.form.get('c_conf'))
    v_type = int(request.form.get('v_type'))
    c_vehs = int(request.form.get('c_vehs'))

    c_wthr = int(request.form.get('c_wthr'))
    c_rcfg = int(request.form.get('c_rcfg'))
    c_raln = int(request.form.get('c_raln'))

    p_sex = int(request.form.get('p_sex'))
    p_psn = int(request.form.get('p_psn'))
    p_user = int(request.form.get('p_user'))
    p_age = int(request.form.get('p_age'))


    df_app = pd.DataFrame({'c_year': [c_year], 'c_mnth': [c_mnth], 'c_wday': [c_wday], 'c_hour': [c_hour], 'c_conf': [c_conf], 'v_type': [v_type],\
                       'c_vehs': [c_vehs], 'c_wthr': [c_wthr], 'c_rcfg': [c_rcfg], 'c_raln': [c_raln], 'p_sex': [p_sex], 'p_psn': [p_psn],\
                       'p_user': [p_user], 'p_age': [p_age]})

    categorical = ['c_mnth', 'c_raln', 'c_rcfg', 'c_wday', 'c_conf', 
               'c_hour', 'c_wthr', 'v_type', 'c_vehs', 'p_psn', 'p_user',  'p_sex']
    numerical = ['c_year', 'p_age']

    df_app_dummies = pd.get_dummies(df_app[categorical])
    df_app = pd.concat([df_app_dummies, df_app[numerical]], axis = 1)

    model_columns = ['c_mnth_1.0','c_mnth_10.0','c_mnth_11.0','c_mnth_12.0','c_mnth_2.0','c_mnth_3.0','c_mnth_4.0','c_mnth_5.0','c_mnth_6.0','c_mnth_7.0','c_mnth_8.0','c_mnth_9.0','c_mnth_UU','c_raln_1','c_raln_2','c_raln_3','c_raln_4','c_raln_5','c_raln_6','c_raln_Q','c_raln_U','c_rcfg_01','c_rcfg_02','c_rcfg_03','c_rcfg_04','c_rcfg_05','c_rcfg_06','c_rcfg_07','c_rcfg_08','c_rcfg_09','c_rcfg_10','c_rcfg_QQ','c_rcfg_UU','c_wday_1.0','c_wday_2.0','c_wday_3.0','c_wday_4.0','c_wday_5.0','c_wday_6.0','c_wday_7.0','c_wday_U','c_conf_01','c_conf_02','c_conf_03','c_conf_04','c_conf_05','c_conf_06','c_conf_21','c_conf_22','c_conf_23','c_conf_24','c_conf_25','c_conf_31','c_conf_32','c_conf_33','c_conf_34','c_conf_35','c_conf_36','c_conf_41','c_conf_QQ','c_conf_UU','c_hour_00','c_hour_01','c_hour_02','c_hour_03','c_hour_04','c_hour_05','c_hour_06','c_hour_07','c_hour_08','c_hour_09','c_hour_10','c_hour_11','c_hour_12','c_hour_13','c_hour_14','c_hour_15','c_hour_16','c_hour_17','c_hour_18','c_hour_19','c_hour_20','c_hour_21','c_hour_22','c_hour_23','c_hour_UU','c_wthr_1','c_wthr_2','c_wthr_3','c_wthr_4','c_wthr_5','c_wthr_6','c_wthr_7','c_wthr_Q','c_wthr_U','v_type_01','v_type_05','v_type_06','v_type_07','v_type_08','v_type_09','v_type_10','v_type_11','v_type_14','v_type_16','v_type_17','v_type_18','v_type_19','v_type_20','v_type_21','v_type_22','v_type_23','v_type_NN','v_type_QQ','v_type_UU','p_psn_11','p_psn_12','p_psn_13','p_psn_21','p_psn_22','p_psn_23','p_psn_31','p_psn_32','p_psn_33','p_psn_96','p_psn_97','p_psn_98','p_psn_99','p_psn_NN','p_psn_QQ','p_psn_UU','p_user_1','p_user_2','p_user_3','p_user_4','p_user_5','p_user_U','p_sex_0','p_sex_1','p_sex_N','p_sex_U','c_vehs_1.0','c_vehs_10.0','c_vehs_11.0','c_vehs_12.0','c_vehs_13.0','c_vehs_14.0','c_vehs_15.0','c_vehs_16.0','c_vehs_17.0','c_vehs_18.0','c_vehs_19.0','c_vehs_2.0','c_vehs_20.0','c_vehs_21.0','c_vehs_22.0','c_vehs_23.0','c_vehs_24.0','c_vehs_25.0','c_vehs_26.0','c_vehs_27.0','c_vehs_28.0','c_vehs_29.0','c_vehs_3.0','c_vehs_30.0','c_vehs_31.0','c_vehs_32.0','c_vehs_33.0','c_vehs_34.0','c_vehs_35.0','c_vehs_36.0','c_vehs_37.0','c_vehs_38.0','c_vehs_39.0','c_vehs_4.0','c_vehs_40.0','c_vehs_41.0','c_vehs_43.0','c_vehs_44.0','c_vehs_46.0','c_vehs_5.0','c_vehs_51.0','c_vehs_54.0','c_vehs_56.0','c_vehs_57.0','c_vehs_58.0','c_vehs_6.0','c_vehs_7.0','c_vehs_71.0','c_vehs_72.0','c_vehs_77.0','c_vehs_8.0','c_vehs_9.0','c_vehs_UU','c_year','p_age']

    def return_not_matches(a, b):
        return [[x for x in a if x not in b], [x for x in b if x not in a]]

    columns_to_add = return_not_matches(df_app.columns, model_columns)[1]

    df_app[columns_to_add] = 0

    df_app = df_app[model_columns]

    final_features = np.array(df_app).reshape(1,-1)

    print('final_features: ')
    print(final_features)

    prediction = model.predict(final_features)

    if prediction == 0:
        output = 'No se esperan fallecidos en el accidente.'
    else: 
        output = 'Previsiblemente hay al menos un fallecido en el accidente.'
    # output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='{}'.format(output))
'''
@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    print('data: ') 
    print(data)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
'''


if __name__ == "__main__":
    app.run(debug=True)