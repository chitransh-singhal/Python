import pkg_resources
import numpy as np

vectorizer = joblib.load(pkg_resources.resource_filename('profanity_check', 'data')) #here data is the data given from twitter
model = joblib.load(pkg_resources.resource_filename('profanity_check', 'data')) #here data is the data given from twitter

def _get_profane_prob(prob):
  return prob[1]

def predict(texts):
  return model.predict(vectorizer.transform(texts))

def predict_prob(texts):
  return np.apply_along_axis(_get_profane_prob, 1, model.predict_proba(vectorizer.transform(texts)))
