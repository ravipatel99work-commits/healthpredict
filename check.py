import joblib

# Load your model
model = joblib.load("model/disease_model.pkl")

# Print the type of model
print("Model type:", type(model))

# Check more details if it’s a scikit-learn model
if hasattr(model, "get_params"):
    print("Parameters:", model.get_params())
