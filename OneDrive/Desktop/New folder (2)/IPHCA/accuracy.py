from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Assuming you have already defined clf3 and inputtest
clf3 = tree.DecisionTreeClassifier()
clf3.fit(X_train, y_train)

inputtest = [0] * len(l1)
for k in range(len(l1)):
    for z in symptoms:
        if z == l1[k]:
            inputtest[k] = 1

inputtest = [inputtest]
predicted = clf3.predict(inputtest)[0]
print("predicted : ", predicted)

# Calculate accuracy
y_pred = clf3.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
