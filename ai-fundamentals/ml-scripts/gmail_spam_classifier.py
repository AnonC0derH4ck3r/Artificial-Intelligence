"""
Gmail-Integrated Spam Classifier
---------------------------------
Pulls real emails from your Gmail account (INBOX + SPAM folders),
uses the folder as the label (auto-labeling instead of hardcoding),
then trains a Naive Bayes classifier on them.

SETUP (one-time):
1. Enable Gmail API in Google Cloud Console.
2. Create OAuth Client ID (Desktop app) -> download as 'credentials.json'
   and place it in the same folder as this script.
3. pip install google-auth google-auth-oauthlib google-api-python-client scikit-learn

FIRST RUN:
A browser window will open asking you to log in and approve access.
This creates a 'token.json' file so you won't have to log in again.
"""

import os
import base64
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Read-only access is all we need
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


# ---------------------------------------------------------
# 1. Authenticate with Gmail
# ---------------------------------------------------------
def get_gmail_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


# ---------------------------------------------------------
# 2. Fetch emails from a given label (INBOX or SPAM)
#    and extract subject + a snippet of the body
# ---------------------------------------------------------
def fetch_emails(service, label, max_results=50):
    """Returns a list of dicts: {'subject': ..., 'text': 'subject + body combined'}"""
    results = service.users().messages().list(
        userId="me", labelIds=[label], maxResults=max_results
    ).execute()

    messages = results.get("messages", [])
    emails = []

    for msg in messages:
        msg_data = service.users().messages().get(
            userId="me", id=msg["id"], format="full"
        ).execute()

        headers = msg_data.get("payload", {}).get("headers", [])
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(no subject)")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "(unknown sender)")
        body = extract_body(msg_data.get("payload", {}))

        # emails.append({"subject": subject, "text": f"{subject} {body}"})
        emails.append({"subject": subject, "sender": sender, "text": f"{subject} {body}"})

    return emails


def extract_body(payload):
    """Recursively pull plain-text body out of a Gmail message payload."""
    if payload.get("mimeType") == "text/plain" and "data" in payload.get("body", {}):
        data = payload["body"]["data"]
        return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")

    for part in payload.get("parts", []):
        text = extract_body(part)
        if text:
            return text

    # fall back to snippet if no plain-text part found (e.g. HTML-only email)
    return ""


# ---------------------------------------------------------
# 3. Build the labeled dataset from your real inbox
# ---------------------------------------------------------
print("Authenticating with Gmail...")
service = get_gmail_service()

print("Fetching NOT SPAM emails from Inbox...")
ham_emails = fetch_emails(service, "INBOX", max_results=50)

print("Fetching SPAM emails from Spam folder...")
spam_emails = fetch_emails(service, "SPAM", max_results=50)

print(f"Collected {len(ham_emails)} ham emails and {len(spam_emails)} spam emails.")

# ---------------------------------------------------------
# 3b. Print out subjects grouped by label
# ---------------------------------------------------------
print(f"\n===== SPAM subjects ({len(spam_emails)}) =====")
# print()
for e in spam_emails:
    print(f"  - From: {e['sender']} | {e['subject']}")

print(f"\n===== NOT SPAM subjects ({len(ham_emails)}) =====")
for e in spam_emails:
    print(f"  - From: {e['sender']} | {e['subject']}")

emails = [e["text"] for e in spam_emails] + [e["text"] for e in ham_emails]
labels = [1] * len(spam_emails) + [0] * len(ham_emails)  # 1 = SPAM, 0 = NOT SPAM

if len(set(labels)) < 2:
    raise SystemExit("Need at least some emails in both Inbox and Spam to train. "
                      "Try again once you have a few spam emails in your Spam folder.")

all_subjects = [e["subject"] for e in spam_emails] + [e["subject"] for e in ham_emails]

# ---------------------------------------------------------
# 4. Train/test split
#    (keep subjects aligned to the same split via indices)
# ---------------------------------------------------------
indices = list(range(len(emails)))
idx_train, idx_test = train_test_split(
    indices, test_size=0.3, random_state=42, stratify=labels
)
X_train = [emails[i] for i in idx_train]
X_test = [emails[i] for i in idx_test]
y_train = [labels[i] for i in idx_train]
y_test = [labels[i] for i in idx_test]
subjects_test = [all_subjects[i] for i in idx_test]

# ---------------------------------------------------------
# 5. Vectorize text (Bag of Words)
# ---------------------------------------------------------
vectorizer = CountVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------------------------------------
# 6. Train Naive Bayes classifier
# ---------------------------------------------------------
model = MultinomialNB(class_prior=[0.5, 0.5])  # force 50/50 prior instead of matching skewed data
model.fit(X_train_vec, y_train)

# ---------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------
y_pred = model.predict(X_test_vec)
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=["NOT SPAM", "SPAM"]))

# ---------------------------------------------------------
# 7b. Print predicted subjects, grouped by predicted label
#     (this is the model's guess on held-out test emails)
# ---------------------------------------------------------
predicted_spam_subjects = [s for s, p in zip(subjects_test, y_pred) if p == 1]
predicted_ham_subjects = [s for s, p in zip(subjects_test, y_pred) if p == 0]

print(f"\n===== PREDICTED SPAM subjects ({len(predicted_spam_subjects)}) =====")
for s in predicted_spam_subjects:
    print(f"  - {s}")

print(f"\n===== PREDICTED NOT SPAM subjects ({len(predicted_ham_subjects)}) =====")
for s in predicted_ham_subjects:
    print(f"  - {s}")


# ---------------------------------------------------------
# 8. Use it to classify the newest unread email in your inbox
# ---------------------------------------------------------
def classify_latest_email(service, model, vectorizer):
    results = service.users().messages().list(
        userId="me", labelIds=["INBOX"], maxResults=1
    ).execute()
    messages = results.get("messages", [])
    if not messages:
        print("No emails found.")
        return

    msg_data = service.users().messages().get(
        userId="me", id=messages[0]["id"], format="full"
    ).execute()

    headers = msg_data.get("payload", {}).get("headers", [])
    subject = next((h["value"] for h in headers if h["name"] == "Subject"), "")
    body = extract_body(msg_data.get("payload", {}))
    text = f"{subject} {body}"

    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    label = "SPAM" if pred == 1 else "NOT SPAM"
    print(f"\nLatest email: \"{subject}\"")
    print(f"Prediction: {label}")


classify_latest_email(service, model, vectorizer)