import xgboost as xgb


# Initialize an empty XGBClassifier
loaded_model = xgb.XGBClassifier()

# Load the trained weights and tree logic
loaded_model.load_model('crisis_prediction_model.json')

