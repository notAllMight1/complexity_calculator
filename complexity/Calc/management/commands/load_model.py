# core/management/commands/load_model.py

from django.core.management.base import BaseCommand
import joblib
from ML.model_training import predict_time_complexity

class Command(BaseCommand):
    help = 'Load the trained model at startup'

    def handle(self, *args, **kwargs):
        # Load model and vectorizer at startup
        self.stdout.write("Loading model and vectorizer...")
        self.ensemble_model = joblib.load('ensemble_model.pkl')
        self.vectorizer = joblib.load('vectorizer.pkl')
        self.stdout.write("Model loaded successfully!")
