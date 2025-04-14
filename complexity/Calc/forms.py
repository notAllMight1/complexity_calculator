from django import forms

class CodeInputForm(forms.Form):
    LANGUAGES = [
        ('python', 'Python'),
        ('java', 'Java'),
        ('cpp', 'C++'),
    ]
    
    code = forms.CharField(widget=forms.Textarea(attrs={'rows': 10, 'cols': 60}), label='Code Snippet')
    language = forms.ChoiceField(choices=LANGUAGES, label='Programming Language')
