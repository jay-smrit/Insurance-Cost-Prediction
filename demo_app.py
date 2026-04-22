import streamlit as st

st.title("Welcome to this demo app!")

st.header("This is a Insurance Cost Prediction app. ")

st.subheader("Please enter the details of your demographic and health condition to get a premium price Prediction")

if st.checkbox("Show/Hide"):
    st.text("This is a demo app to show how to use Streamlit to create a simple web application. It is designed to predict insurance costs based on user input. ")

if st.button("Click here for more info:"):
    st.text("This is a Insurance premium price app")

def sqr(num):
    return num ** 2

num = st.number_input("Enter a number to square:", value=0)
if st.button("Calculate Square"):
    result = sqr(num)
    st.write(f"The square of {num} is {result}.")