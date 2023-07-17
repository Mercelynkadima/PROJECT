from sklearn import naive_bayes, svm
from sklearn.base import is_classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import joblib

# Assuming you have trained models named svm_model, logreg_model, dt_model, nb_model, rf_model

# Save SVM model
joblib.dump(svm, "model_svm.pkl")

# Save Logistic Regression model
joblib.dump(LogisticRegression, "model_logreg.pkl")

# Save Decision Tree model
joblib.dump(DecisionTreeClassifier, "model_dt.pkl")

# Save Naive Bayes model
joblib.dump(naive_bayes, "model_nb.pkl")

# Save Random Forest model
joblib.dump(RandomForestClassifier, "model_rf.pkl")


