from django.db import models
from django.contrib.auth.models import User

class Prediction(models.Model):
    # Link to the user who made the prediction (optional)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)

    # Code snippet input by the user
    code = models.TextField()

    # Language of the code snippet
    language = models.CharField(max_length=10, choices=[('python', 'Python'), ('java', 'Java'), ('cpp', 'C++')])

    # Predicted time complexity (e.g., O(n log n), O(n^2), etc.)
    predicted_complexity = models.CharField(max_length=50)

    # Date and time of the prediction request
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction for {self.language} code: {self.predicted_complexity}"

    class Meta:
        ordering = ['-created_at']

class Dataset(models.Model):
    """
    A model to manage datasets uploaded by admin, such as for training or retraining the ML model.
    """
    name = models.CharField(max_length=100)  # Name of the dataset (e.g., 'CodeNet Subset')
    file = models.FileField(upload_to='datasets/')  # File field to store the dataset CSV/JSON
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ['-created_at']

# Optionally, you could extend the Django User model if you need custom user-related fields.
# However, Django's default User model should suffice for most use cases.
