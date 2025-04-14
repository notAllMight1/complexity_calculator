import os
from django.shortcuts import render
from django.http import JsonResponse
from .forms import CodeInputForm
from ML.model_training import predict_time_complexity

# Get BASE_DIR for consistent file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'ensemble_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')

def index(request):
    predicted_complexity = None
    if request.method == 'POST':
        form = CodeInputForm(request.POST)
        if form.is_valid():
            code = form.cleaned_data['code']
            language = form.cleaned_data['language']
            # Predict using model
            predicted_complexity = predict_time_complexity(
                code,
                language,
                model_path=MODEL_PATH,
                vectorizer_path=VECTORIZER_PATH
            )
    else:
        form = CodeInputForm()

    return render(request, 'index.html', {'form': form, 'predicted_complexity': predicted_complexity})

def predict(request):
    if request.method == 'POST':
        code = request.POST.get('code', '')
        language = request.POST.get('language', '')
        if code and language:
            predicted_complexity = predict_time_complexity(
                code,
                language,
                model_path=MODEL_PATH,
                vectorizer_path=VECTORIZER_PATH
            )
            return JsonResponse({'predicted_complexity': predicted_complexity})
        else:
            return JsonResponse({'error': 'Invalid input'}, status=400)
    return JsonResponse({'error': 'POST method required'}, status=400)
