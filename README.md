🏦 Loan Approval Predictor
A machine learning web application that predicts loan approval outcomes using a Naive Bayes classifier built with Python, scikit-learn, and Streamlit.

🤖 How the ML Model Works
Algorithm: Naive Bayes
The model uses the Naive Bayes algorithm (a probabilistic classifier based on Bayes   Theorem with an assumption of feature independence).

Bayes Theorem:
P(Approved | Features) = P(Features | Approved) × P(Approved) / P(Features)
In plain terms: the model calculates the probability that they belong to the "Approved" or "Rejected" class, and picks the one with the higher probability.


ML PIPELINE  👇 

Raw Input
    ↓
Preprocessing (Label Encoding / Scaling)
    ↓
Naive Bayes Classifier
    ↓
Probability Score
    ↓
Approved ✅ / Rejected ❌

👤 Author
Aditya
Built with ❤️ using self-trained ML + Streamlit
