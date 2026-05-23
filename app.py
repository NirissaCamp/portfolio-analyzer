"""Portfolio Analyzer - Hello World smoke test.

This file is just to verify the dev environment works.
We'll replace it with the real app in the next phase.
"""

import streamlit as st


def main():
    st.title("Portfolio Analyzer")
    st.subheader("Hello World - environment check")

    st.write("If you can see this in the browser, your setup works!")

    st.markdown("---")

    name = st.text_input("What's your name?", value="Narissa")

    if st.button("Greet me"):
        st.success(f"Hello, {name}! Welcome to your Python journey.")

    st.markdown("---")
    st.caption("Built with Streamlit + Python 3.14")


if __name__ == "__main__":
    main()
