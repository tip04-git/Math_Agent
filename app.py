import streamlit as st
from math_solver import agentic_answer

st.title("Math Professor Agent 🤖")
st.write("Ask any math question and get a step-by-step solution!")

question = st.text_input("Enter your math question:")

if st.button("Get Solution"):
    if question.strip():
        with st.spinner("Solving..."):
            answer = agentic_answer(question)
        st.markdown("### Step-by-step solution:")
        st.write(answer)
    else:
        st.warning("Please enter a question.")