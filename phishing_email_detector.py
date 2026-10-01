#!/usr/bin/env python3
"""
Phishing Email Detection Model

Educational ML project using Scikit-learn to classify emails as:
    0 -> Safe
    1 -> Phishing

The model uses:
- Email text (subject + body)
- URL count
- Suspicious keyword count
- Exclamation mark count
- Message length

Usage:
    python phishing_email_detector.py

Optional:
    python phishing_email_detector.py --dataset emails.csv

CSV format:
    text,label
    "Your account has been updated",Safe
    "Urgent! Verify your account at http://...",Phishing
"""

import argparse
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression


# A small built-in dataset lets the project run immediately.
# Replace it with a larger real dataset for better performance.
DEMO_DATA = [
    ("Your monthly bank statement is ready. You can view it in your secure account.", "Safe"),
    ("Meeting reminder: project review is scheduled for tomorrow at 10 AM.", "Safe"),
    ("Your order has been shipped and will arrive on Friday.", "Safe"),
    ("Thank you for registering. Your account settings are available in the portal.", "Safe"),
    ("Your password was changed successfully. Contact support if this was not you.", "Safe"),
    ("The team meeting has been moved to 3 PM. Please check the calendar.", "Safe"),
    ("Your invoice for this month is available in the customer portal.", "Safe"),
    ("Welcome to our newsletter. Here are this week's company updates.", "Safe"),
    ("Your university examination timetable is now available.", "Safe"),
    ("The HR team has shared a new document in the employee portal.", "Safe"),
    ("Your payment was received successfully. Thank you for your purchase.", "Safe"),
    ("Please review the attached project report before the meeting.", "Safe"),
    ("Your flight booking confirmation is available in your travel account.", "Safe"),
    ("The library book you reserved is now ready for collection.", "Safe"),
    ("Your subscription renewal was completed successfully.", "Safe"),
    ("Please confirm your attendance for the workshop.", "Safe"),
    ("Your support ticket has been updated by our customer service team.", "Safe"),
    ("The weekly team report has been uploaded to the shared drive.", "Safe"),
    ("Your application has been received. We will contact you with updates.", "Safe"),
    ("Reminder: your appointment is scheduled for Monday afternoon.", "Safe"),

    ("URGENT! Your account will be suspended. Verify your password immediately at http://secure-login.example.com", "Phishing"),
    ("Congratulations! You won a prize. Click http://claim-prize.example.com now to receive your reward.", "Phishing"),
    ("Your bank account is locked. Confirm your credentials at http://verify-account.example.com immediately.", "Phishing"),
    ("FINAL WARNING! Send your login details to avoid account closure.", "Phishing"),
    ("You have received a refund. Click this link to claim it now: http://refund.example.com", "Phishing"),
    ("Security alert! Your account has been compromised. Verify your identity at http://login.example.com", "Phishing"),
    ("Act now!!! Your password expires today. Click http://password-reset.example.com to continue.", "Phishing"),
    ("Dear customer, update your banking information immediately using http://bank-update.example.com", "Phishing"),
    ("You have been selected for a cash reward. Send your details to claim the money.", "Phishing"),
    ("Your email storage is full. Verify your account at http://mail-check.example.com or it will be closed.", "Phishing"),
    ("URGENT: confirm your username and password at http://account-security.example.com", "Phishing"),
    ("Your package could not be delivered. Pay the small fee at http://delivery-fee.example.com", "Phishing"),
    ("Click here immediately to prevent your account from being deleted: http://account-alert.example.com", "Phishing"),
    ("You won a gift card! Claim it now by entering your personal information.", "Phishing"),
    ("Important security notice: verify your account credentials at http://secure-verify.example.com", "Phishing"),
    ("Your tax refund is waiting. Submit your bank information through http://tax-refund.example.com", "Phishing"),
    ("Your account has unusual activity. Login now at http://account-check.example.com to secure it.", "Phishing"),
    ("FINAL NOTICE!!! Your service will be terminated unless you confirm your details.", "Phishing"),
    ("Free reward waiting for you. Click http://free-gift.example.com and enter your details.", "Phishing"),
    ("Your online account needs verification. Provide your password at http://verify-user.example.com", "Phishing"),
]


SUSPICIOUS_KEYWORDS = [
    "urgent",
    "verify",
    "verification",
    "password",
    "credential",
    "credentials",
    "account suspended",
    "account locked",
    "click here",
    "click",
    "confirm",
    "immediately",
    "final warning",
    "security alert",
    "winner",
    "won",
    "prize",
    "reward",
    "refund",
    "bank",
    "login",
    "payment",
    "claim",
    "limited time",
    "act now",
    "gift card",
]


def create_demo_dataframe():
    return pd.DataFrame(DEMO_DATA, columns=["text", "label"])


def load_dataset(path=None):
    if path:
        df = pd.read_csv(path)

        # Accept common column names.
        text_column = None
        label_column = None

        for name in ["text", "email", "message", "body", "content"]:
            if name in df.columns:
                text_column = name
                break

        for name in ["label", "class", "category", "target"]:
            if name in df.columns:
                label_column = name
                break

        if text_column is None or label_column is None:
            raise ValueError(
                "CSV must contain a text column (text/email/message/body/content) "
                "and a label column (label/class/category/target)."
            )

        df = df[[text_column, label_column]].copy()
        df.columns = ["text", "label"]
    else:
        df = create_demo_dataframe()

    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    label_map = {
        "0": "Safe",
        "1": "Phishing",
        "safe": "Safe",
        "legitimate": "Safe",
        "ham": "Safe",
        "normal": "Safe",
        "phishing": "Phishing",
        "spam": "Phishing",
    }

    df["label"] = df["label"].map(
        lambda value: label_map.get(value, value.title())
    )

    valid_labels = {"Safe", "Phishing"}
    df = df[df["label"].isin(valid_labels)].reset_index(drop=True)

    if len(df) < 10 or df["label"].nunique() < 2:
        raise ValueError("Dataset must contain at least 10 rows and both Safe and Phishing labels.")

    return df


def url_count(text):
    return len(re.findall(r"https?://\S+|www\.\S+", str(text), flags=re.IGNORECASE))


def keyword_count(text):
    lower = str(text).lower()
    return sum(lower.count(keyword) for keyword in SUSPICIOUS_KEYWORDS)


def exclamation_count(text):
    return str(text).count("!")


def text_length(text):
    return len(str(text))


def add_engineered_features(df):
    result = df.copy()
    result["url_count"] = result["text"].apply(url_count)
    result["keyword_count"] = result["text"].apply(keyword_count)
    result["exclamation_count"] = result["text"].apply(exclamation_count)
    result["text_length"] = result["text"].apply(text_length)
    return result


def build_model():
    """
    Combine TF-IDF text features with manually extracted URL/keyword features.
    """
    text_features = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000,
    )

    numeric_features = FunctionTransformer(
        lambda x: x,
        validate=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_features, "text"),
            (
                "numeric",
                numeric_features,
                ["url_count", "keyword_count", "exclamation_count", "text_length"],
            ),
        ]
    )

    from sklearn.pipeline import Pipeline

    model = Pipeline(
        steps=[
            ("features", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    return model


def plot_confusion_matrix(y_test, y_pred, output_file="confusion_matrix.png"):
    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=["Safe", "Phishing"],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Safe", "Phishing"],
    )

    display.plot(values_format="d")
    plt.title("Phishing Email Detection - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.show()


def predict_email(model, email_text):
    prediction = model.predict(
        add_engineered_features(pd.DataFrame({"text": [email_text]}))
    )[0]

    probabilities = model.predict_proba(
        add_engineered_features(pd.DataFrame({"text": [email_text]}))
    )[0]

    classes = model.named_steps["classifier"].classes_
    confidence = probabilities[list(classes).index(prediction)]

    return prediction, confidence


def main():
    parser = argparse.ArgumentParser(
        description="Train a Scikit-learn phishing email detection model."
    )
    parser.add_argument(
        "--dataset",
        help="Path to CSV dataset. If omitted, a small built-in demo dataset is used.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction used for testing (default: 0.25).",
    )

    args = parser.parse_args()

    print("=" * 65)
    print("PHISHING EMAIL DETECTION MODEL")
    print("=" * 65)

    try:
        df = load_dataset(args.dataset)
    except Exception as error:
        print(f"Dataset error: {error}")
        return

    df = add_engineered_features(df)

    print(f"\nDataset size: {len(df)}")
    print("\nClass distribution:")
    print(df["label"].value_counts())

    X = df[
        [
            "text",
            "url_count",
            "keyword_count",
            "exclamation_count",
            "text_length",
        ]
    ]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    model = build_model()

    print("\nTraining model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 65)
    print("MODEL RESULTS")
    print("=" * 65)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Creating confusion matrix...")
    plot_confusion_matrix(y_test, y_pred)

    # Test a few example emails after training.
    examples = [
        "Your meeting is confirmed for tomorrow at 10 AM.",
        "URGENT! Verify your account password immediately at http://fake-login.example.com",
    ]

    print("\nExample Predictions:")
    print("-" * 65)

    for email in examples:
        prediction, confidence = predict_email(model, email)
        print(f"Email      : {email}")
        print(f"Prediction : {prediction}")
        print(f"Confidence : {confidence * 100:.2f}%")
        print()

    print("Done.")
    print("Confusion matrix saved as: confusion_matrix.png")


if __name__ == "__main__":
    main()
