#!/usr/bin/python3
import joblib

model = joblib.load('/root/filter_plugins/marks_prediction.pk1')

class FilterModule(object):
    def filters(self):
        return {
            'predict_marks': self.predict_marks,
        }

    def predict_marks(self, hrs_study, gender, course_status):
        if gender == 'male':
            g = 1
        else:
            g = 0

        if course_status == 'completed':
            cs = 1
        else:
            cs = 0

        pred = [[hrs_study, g, cs]]

        prediction = model.predict(pred)
        prediction = prediction[0]

        return prediction
