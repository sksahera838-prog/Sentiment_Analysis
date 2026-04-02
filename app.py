import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

st.title("📊 Sentiment Analysis Dashboard")


st.sidebar.header("Options")
mode = st.sidebar.radio("Choose Mode", ["Single Text", "CSV Upload"])


if mode == "Single Text":
    st.subheader("Analyze Single Text")

    user_input = st.text_area("Enter text:")

    if st.button("Analyze"):
        if user_input.strip() == "":
            st.warning("Please enter text")
        else:
            X = vectorizer.transform([user_input])
            pred = model.predict(X)[0]
            prob = model.predict_proba(X).max()

            if pred == "positive":
                st.success(f"Positive 😊 ({prob:.2f})")
            elif pred == "negative":
                st.error(f"Negative 😡 ({prob:.2f})")
            else:
                st.info(f"Neutral 😐 ({prob:.2f})")
elif mode == "CSV Upload":
    st.subheader("Upload CSV File")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("### Preview Data")
        st.dataframe(df.head())

        
        text_column = st.selectbox("Select Text Column", df.columns)

        if st.button("Analyze CSV"):
            texts = df[text_column].astype(str)

            X = vectorizer.transform(texts)
            predictions = model.predict(X)

            df["Sentiment"] = predictions

            st.write("### Results")
            st.dataframe(df)

            
            st.write("### 📊 Sentiment Distribution")

            sentiment_counts = df["Sentiment"].value_counts()

            fig, ax = plt.subplots()
            ax.bar(sentiment_counts.index, sentiment_counts.values)
            ax.set_title("Sentiment Count")
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Count")

            st.pyplot(fig)

            
            st.write("### 🥧 Sentiment Pie Chart")
            fig2, ax2 = plt.subplots()
            ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
            ax2.axis('equal')

            st.pyplot(fig2)

            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results",
                data=csv,
                file_name="sentiment_results.csv",
                mime="text/csv"
            )