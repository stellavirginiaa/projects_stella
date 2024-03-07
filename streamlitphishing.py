import streamlit as st
import pandas as pd
from urllib.parse import urlparse
import joblib
import phishing_project

# Load the trained model
@st.cache_data
def load_model():
    return joblib.load('m1_rf_mean_acc87p.joblib')

def output_link(prediction, url1):
    if prediction == 0:
        st.write(f'<a href="{url1}">{url1}</a>', unsafe_allow_html=True)
        st.write('<p style="color:green;">You can click the link above. It is safe!</p>', unsafe_allow_html=True)
    elif prediction == 1:
        st.write('<p style="color:red;">WARNING!! It is suspicious</p>', unsafe_allow_html=True)
    elif prediction == -1:
        st.write("<p>It's not URL at all</p>", unsafe_allow_html=True)
    else:
        st.write('Input not found')

def main():

    st.write("""

    Phishing, a deceitful tactic employed by cybercriminals, poses a significant threat to online security. It involves the malicious attempt to extract sensitive information like passwords or financial data by impersonating reputable entities through electronic means such as emails, text messages, or counterfeit websites.

    In the realm of cybersecurity, raising awareness about identifying phishing links is paramount to safeguarding personal information from cyber attacks. Leveraging machine learning, our project utilizes the Random Forest algorithm to analyze URLs and assess their security status. With an impressive accuracy rate of 83.3%, this tool empowers users to verify the safety of URLs, thereby enhancing their digital security posture. Simply input the URL you wish to examine and click 'Detect' to unveil the results. Protect yourself from cyber threats with our innovative solution!
    """)
    # Input URLs
    url1 = st.text_input('Enter URL:')

    if st.button('Detect'):
        # Load model
        model = load_model()

        f1 = [phishing_project.calculate_www(url1)]
        f2 = [phishing_project.calculate_com(url1)]
        f3 = [phishing_project.calculate_dot(url1)]
        f4 = [phishing_project.calculate_slash(url1)]
        f5 = [phishing_project.count_digits(url1)]
        f6 = [len(url1)]
        f7 = [phishing_project.calculate_hostname_length(url1)]
        f8 = [phishing_project.calculate_ratio_digits(url1)]
        f9 = [phishing_project.calculate_ratio_digits(urlparse(url1).netloc)]

        new_data = pd.DataFrame({'nb_www': f1, 'nb_com': f2, 'length_dot': f3, 'length_slash': f4, 'length_digits': f5, 'length_url': f6, 'length_hostname': f7, 'ratio_digits_url': f8,
                         'ratio_digits_host': f9})

        # Make predictions
        predictions_rfr = phishing_project.clfRFR.predict(new_data)

        # Output link
        output_link(predictions_rfr[0], url1)

if __name__ == '__main__':
    main()
