import pandas as pd
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast',
                'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
                    'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
                 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
             'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
                   'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)
total = len(df)
yes_count = len(df[df['PlayTennis'] == 'Yes'])
no_count = len(df[df['PlayTennis'] == 'No'])

P_yes = yes_count / total
P_no = no_count / total

print(f"P(Yes) = {P_yes:.3f}, P(No) = {P_no:.3f}")

def calc_conditional_prob(df, feature, value, target):
    subset = df[df['PlayTennis'] == target]
    count = len(subset[subset[feature] == value])
    total = len(subset)
    return (count + 1) / (total + len(df[feature].unique()))


def predict(sample):  
    # Likelihood for "Yes"
    prob_yes = P_yes
    for feature, value in sample.items():
        prob_yes *= calc_conditional_prob(df, feature, value, "Yes")
    
    # Likelihood for "No"
    prob_no = P_no
    for feature, value in sample.items():
        prob_no *= calc_conditional_prob(df, feature, value, "No")
    
    print(f"\nLikelihood(Yes) = {prob_yes:.6f}")
    print(f"Likelihood(No)  = {prob_no:.6f}")
    
    return "Yes" if prob_yes > prob_no else "No"
sample = {"Outlook":"Sunny", "Temperature":"Cool", "Humidity":"High", "Wind":"Strong"}
prediction = predict(sample)
print("\nFinal Prediction for sample:", sample, "→", prediction)