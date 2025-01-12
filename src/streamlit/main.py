import streamlit as st


BASE_URL = "https://fastapi:8000"


def main():

    st.set_page_config(page_title="Main")

    st.write(
        """
        Streamlit app for the Watch Pricing project.
        """
    )


if __name__ == "__main__":
    main()
