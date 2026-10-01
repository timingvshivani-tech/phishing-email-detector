# Phishing Email Detection Model

A simple **Machine Learning mini project** that uses **Scikit-learn** to classify emails as **Phishing** or **Safe**.

The project combines:

- TF-IDF text features
- URL detection
- Suspicious keyword counting
- Exclamation mark counting
- Email text length
- Logistic Regression classification
- Accuracy measurement
- Classification report
- Confusion matrix

## Project Structure

```text
phishing-email-detection/
│
├── phishing_email_detector.py
└── README.md
```

When the program runs, it also creates:

```text
confusion_matrix.png
```

## Requirements

Python 3.8+ is recommended.

Install the required packages:

```bash
pip install pandas scikit-learn matplotlib
```

## Run the Project

### Option 1: Run with the built-in demo dataset

No dataset file is required.

```bash
python phishing_email_detector.py
```

The program contains a small educational dataset so that you can run and demonstrate the project immediately.

### Option 2: Use your own CSV dataset

```bash
python phishing_email_detector.py --dataset emails.csv
```

The CSV should contain two columns:

```csv
text,label
"Your meeting is scheduled for tomorrow",Safe
"URGENT! Verify your account at http://example.com",Phishing
```

The program also recognizes common alternative column names such as:

- Text: `text`, `email`, `message`, `body`, `content`
- Label: `label`, `class`, `category`, `target`

Recognized labels include:

- `Safe`
- `Legitimate`
- `Ham`
- `Phishing`
- `Spam`

## How It Works

### 1. Dataset

The model is trained using email messages labelled as either:

```text
Safe
Phishing
```

For a real project, use a substantially larger and representative dataset rather than the small built-in demo data.

### 2. Text Feature Extraction

The email text is converted into numerical features using **TF-IDF**.

TF-IDF gives higher importance to words that are useful for distinguishing one class from another.

The model uses:

- Single words
- Two-word combinations

For example:

```text
"verify account"
"urgent password"
"meeting tomorrow"
```

### 3. URL Feature

The program counts URLs such as:

```text
http://example.com
https://example.com
www.example.com
```

Phishing messages often contain suspicious links, so URL count is included as an additional feature.

### 4. Suspicious Keyword Feature

The program checks for keywords and phrases such as:

```text
urgent
verify
password
credentials
account suspended
click here
immediately
final warning
security alert
prize
reward
claim
```

This is a simple educational feature and should not be treated as proof that an email is malicious.

### 5. Classification

The extracted features are given to a **Logistic Regression** classifier.

The output is:

```text
Safe
```

or

```text
Phishing
```

### 6. Accuracy

The test data is separated from the training data.

The program displays:

```text
Accuracy: XX.XX%
```

Accuracy is calculated as:

```text
correct predictions / total predictions
```

### 7. Confusion Matrix

The program generates:

```text
confusion_matrix.png
```

The matrix shows:

| Actual | Predicted Safe | Predicted Phishing |
|---|---:|---:|
| Safe | True Negative | False Positive |
| Phishing | False Negative | True Positive |

This is useful because accuracy alone does not show which type of mistake the model makes.

## Example Output

```text
=================================================================
PHISHING EMAIL DETECTION MODEL
=================================================================

Dataset size: 40

Class distribution:
Safe        20
Phishing    20

Training model...

=================================================================
MODEL RESULTS
=================================================================
Accuracy: XX.XX%

Classification Report:
              precision    recall  f1-score   support

    Phishing       ...
        Safe       ...

Creating confusion matrix...

Example Predictions:
-----------------------------------------------------------------
Email      : Your meeting is confirmed for tomorrow at 10 AM.
Prediction : Safe
Confidence : XX.XX%

Email      : URGENT! Verify your account password immediately at ...
Prediction : Phishing
Confidence : XX.XX%

Done.
Confusion matrix saved as: confusion_matrix.png
```

The exact accuracy depends on the dataset and train/test split.

## Technologies Used

- **Python**
- **Pandas**
- **Scikit-learn**
- **Matplotlib**
- TF-IDF Vectorization
- Logistic Regression
- Confusion Matrix

## Machine Learning Workflow

```text
Email Dataset
      ↓
Data Cleaning
      ↓
Feature Extraction
      ↓
TF-IDF + URL + Keyword Features
      ↓
Train/Test Split
      ↓
Logistic Regression
      ↓
Prediction
      ↓
Accuracy + Classification Report
      ↓
Confusion Matrix
```

## Important Note About Accuracy

The built-in dataset is intentionally small and is only for demonstrating the project.

A high accuracy on a small or artificially constructed dataset does **not** mean that the model will perform equally well on real-world phishing emails.

For a stronger project:

1. Use a larger labelled dataset.
2. Keep training and testing data separate.
3. Include diverse legitimate emails.
4. Include different phishing techniques.
5. Check precision, recall, and F1-score in addition to accuracy.
6. Test the model on previously unseen emails.

## Limitations

This is an educational phishing detection model. It does not:

- Open or visit suspicious URLs
- Execute attachments
- Inspect email headers such as SPF/DKIM/DMARC
- Perform malware analysis
- Guarantee that an email is safe
- Replace enterprise email security systems

A real email security solution would normally use many additional signals.

## Ethical Use

Use this project for educational, defensive, and authorized security research.

Do not use it to collect credentials, impersonate organizations, distribute phishing emails, or interact with malicious infrastructure.

## Learning Outcomes

This project demonstrates:

- Basic machine learning classification
- Text preprocessing
- TF-IDF feature extraction
- Feature engineering
- Train/test splitting
- Logistic Regression
- Model evaluation
- Accuracy calculation
- Classification reports
- Confusion matrices
- Basic phishing detection concepts

## GitHub Upload

After creating the project folder:

```text
phishing-email-detection/
├── phishing_email_detector.py
└── README.md
```

Run:

```bash
git init
git add .
git commit -m "Add phishing email detection model"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your own GitHub repository URL.
