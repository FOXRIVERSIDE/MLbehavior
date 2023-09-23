import coremltools
import pickle

# Load the Pickle model
with open('activity_classifier_knn.pkl', 'rb') as file:
    model = pickle.load(file)

# Convert the model to CoreML format
coreml_model = coremltools.converters.sklearn.convert(model)

# Save the CoreML model
coreml_model.save('model.mlmodel')