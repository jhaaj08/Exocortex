from django import forms
from .models import Folder, PDFDocument
import os

class FolderInputForm(forms.ModelForm):
    """Form for users to input folder path containing markdown files"""
    
    class Meta:
        model = Folder
        fields = ['name', 'path']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter a name for this folder (e.g., "Python Notes")',
                'required': True
            }),
            'path': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter the full path to your markdown folder (e.g., /Users/john/notes)',
                'required': True
            })
        }
        help_texts = {
            'name': 'Give your folder collection a descriptive name',
            'path': 'Full path to the folder containing .md files'
        }
    
    def clean_path(self):
        """Validate that the path exists and contains markdown files"""
        path = self.cleaned_data['path']
        
        # Check if path exists
        if not os.path.exists(path):
            raise forms.ValidationError("The specified path does not exist.")
        
        # Check if it's a directory
        if not os.path.isdir(path):
            raise forms.ValidationError("The specified path is not a directory.")
        
        # Check if directory contains any markdown files recursively
        markdown_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('.md', '.markdown')):
                    markdown_files.append(file)

        if not markdown_files:
            raise forms.ValidationError("No markdown files (.md or .markdown) found in the specified directory or its subdirectories.")
        
        # Store the count for later use
        self.markdown_file_count = len(markdown_files)
        return path

class PDFUploadForm(forms.ModelForm):
    class Meta:
        #we have defined the PDFDocument model in models.py
        model = PDFDocument
        #these fields are the ones that will be displayed in the form
        fields = ['name', 'pdf_file']
        #widgets are used to style the form fields, name to have textInput and pdf_file to have FileInput
        widgets = {
            'name': forms.TextInput(attrs={
                #classes are css styles
                'class': 'form-control',
                'placeholder': 'Auto-detected from filename'
            }),
            'pdf_file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.pdf'
            })
        }
    
    #this is a constructor for the form
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ✅ Make name field optional
        self.fields['name'].required = False
    
    def clean_pdf_file(self):
        #cleaned_data is a dictionary of the form data, Django creates this for us
        pdf_file = self.cleaned_data.get('pdf_file')
        if pdf_file:
            if not pdf_file.name.lower().endswith('.pdf'):
                raise forms.ValidationError('Please upload a PDF file.')
            if pdf_file.size > 50 * 1024 * 1024:  # 50MB limit
                raise forms.ValidationError('File size must be less than 50MB.')
        return pdf_file
    
    def clean_name(self):
        name = self.cleaned_data.get('name')
        pdf_file = self.cleaned_data.get('pdf_file')
        
        # ✅ Auto-generate name from filename if not provided
        if not name and pdf_file:
            name = pdf_file.name
            if name.endswith('.pdf'):
                name = name[:-4]  # Remove .pdf extension
        
        return name

class QuizImageUploadForm(forms.Form):
    """Form for uploading quiz images for text extraction"""
    
    # Quiz set information
    quiz_name = forms.CharField(
        max_length=255,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter quiz name (e.g., "Biology Chapter 5 Quiz")'
        }),
        help_text='Give your quiz set a descriptive name'
    )
    
    description = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Optional description about this quiz...'
        }),
        help_text='Optional description of the quiz content'
    )
    
    # Image upload - Single file (users can upload multiple sets)
    image = forms.FileField(
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': 'image/*'
        }),
        help_text='Upload an image containing quiz questions'
    )
    
    # Processing options
    source_pdf = forms.ModelChoiceField(
        queryset=PDFDocument.objects.filter(is_duplicate=False),
        required=False,
        empty_label="Select related PDF (optional)",
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text='Link this quiz to a specific PDF document'
    )
    
    auto_categorize = forms.BooleanField(
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        help_text='Automatically categorize difficulty and topics using AI'
    )
    
    def clean_image(self):
        """Validate uploaded image"""
        image = self.cleaned_data.get('image')
        if image:
            if not image.content_type.startswith('image/'):
                raise forms.ValidationError('Please upload a valid image file.')
            if image.size > 10 * 1024 * 1024:  # 10MB limit
                raise forms.ValidationError('File size must be less than 10MB.')
        return image