import pickle
import streamlit as st
import sklearn
# loading the trained model
pickle_in = open('classifier.pkl', 'rb')

classifier = pickle.load(pickle_in)


@st.cache()
# defining the function which will make the prediction using the data which the user inputs
def prediction(CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary,
               Geography_France, Geography_Germany, Geography_Spain):
    # Pre-processing user input
    if Gender == "Male":
        Gender = 0
    else:
        Gender = 1

    if HasCrCard == "No":
        HasCrCard = 0
    else:
        HasCrCard = 1

    if IsActiveMember == "No":
        IsActiveMember = 0
    else:
        IsActiveMember = 1

    if Geography_France == "Yes":
        Geography_France = 1
    else:
        Geography_France = 0

    if Geography_Spain == "Yes":
        Geography_Spain = 1
    else:
        Geography_Spain = 0

    if Geography_Germany == "Yes":
        Geography_Germany = 1
    else:
        Geography_Germany = 0

    # Making predictions
    prediction = classifier.predict(
        [[CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary,
          Geography_France, Geography_Germany, Geography_Spain]])

    if prediction == 0:
        pred = 'Stay!'
    else:
        pred = 'Leave!'
    return pred


# this is the main function in which we define our webpage
def main():
    # front end elements of the web page
    html_temp = """ 
    <div style ="background-color:blue;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Bank Customer Churn Prediction ML App</h1> 
    </div> 
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)

    # following lines create boxes in which user can enter data required to make prediction
    Gender = st.selectbox("Customer's Gender", ("Male", "Female"))
    Age = st.number_input("Customer's Age")
    NumOfProducts = st.selectbox("Total Number of Bank Product The Customer Uses", ("1", "2", "3", "4"))
    Tenure = st.selectbox("Number of Years The Customer Has Been a Client",
                          ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"))
    HasCrCard = st.selectbox('Does The Customer has a Credit Card?', ("Yes", "No"))
    IsActiveMember = st.selectbox('Is The Customer an Active Member?', ("Yes", "No"))
    EstimatedSalary = st.number_input("Estimated Salary of Customer")
    Balance = st.number_input("Customer's Account Balance")
    CreditScore = st.number_input("Customer's Credit Score")
    Geography_France = st.selectbox('Is the Customer From France?', ("Yes", "No"))
    Geography_Spain = st.selectbox('Is the Customer From Spain?', ("Yes", "No"))
    Geography_Germany = st.selectbox('Is the Customer From Germany?', ("Yes", "No"))

    result = ""

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        result = prediction(CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember,
                            EstimatedSalary, Geography_France, Geography_Germany, Geography_Spain)
        st.success('The Customer will {}'.format(result))
        # print(LoanAmount)


if __name__ == '__main__':
    main()