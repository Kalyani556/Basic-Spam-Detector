# Importing tools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Step 1: Example messages
texts = [
    "Win money now",
    "Free prize available",
    "Hello friend how are you",
    "Let's meet tomorrow"
    ]

# Step 2: Labels (spam or not)
labels = ["spam", "spam", "not spam", "not spam"]

# Step 3: Convert words into numbers
vector = CountVectorizer()
X = vector.fit_transform(texts)

# Step 4: Train the model
model = MultinomialNB()
model.fit(X, labels)

# Step 5: Test it
test = ["Free money offer",
    "Are we meeting today?",
    "You are a lucky winner!",
    "Let's go for lunch"]

X_test = vector.transform(test)
result = model.predict(X_test)

# Step 6: Print results
for i in range(len(test)):
    print(test[i], "->", result[i])


